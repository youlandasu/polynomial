import random
import numpy as np
import os
import sys
from tqdm import tqdm
import argparse
import logging
import pickle
import json
from argparse import Namespace

import sys
import numpy as np
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, DataDict, sentence2tensor, collate_data
from model import Seq2Seq, Encoder, Decoder, Attention, loss_fn, metrics
from utils import save_checkpoint, load_checkpoint, save_dict_to_json, save_args, pred2sentence

logger = logging.getLogger(__name__)
torch.cuda.empty_cache() 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions

def process_data(src, tgt):
    '''Create language dictionary for inputs and ground truth.
    Args:
        src: list of input sentences.
        tgt: list of output sentences.
    Return:
        lang_src: DataDict object of source inputs.
        lang_tgt: DataDict object of target outputs.
    '''
    assert len(src) == len(tgt)
    lang_src = DataDict()
    lang_tgt = DataDict()
    for i in range(len(src)):
        lang_src.addSentence(src[i])
        lang_tgt.addSentence(tgt[i])
    return lang_src, lang_tgt

def load_data(data_path, split, batch_size):
    '''Create batches of data for training and evaluation.
    Args:
        data_path: processing data file path
        split: can be a string from "train", "dev" or "test"
        batch_size: size of batch processing
    Return:
        dataset: an iterable data loader that can be directly used as "for batch in dataset".
        where each batch is a Long Tensor on device of dimension [batch_size, seq_len].
        data_size: the total num of examples in the dataset.
        lang_src: DataDict object of source inputs.
        lang_tgt: DataDict object of target outputs.
    '''
    # Create input dataset and datadict
    logger.info("Loading the %s file...", split)
    data_src, data_tgt = load_file(data_path)
    logger.info("Total number of %s sequences: %s", split, len(data_src))
    if split == "train":
        logging.info("Creating the %s dictionary...", split)
        lang_src, lang_tgt = process_data(data_src, data_tgt)

        #Save language dictionary.
        with open("lang_src.pickle","wb") as f:
            pickle.dump(lang_src,f)
        with open("lang_tgt.pickle","wb") as f:
            pickle.dump(lang_tgt,f)
    else:
        with open("lang_src.pickle","rb") as f:
            lang_src = pickle.load(f)
        with open("lang_tgt.pickle","rb") as f:
            lang_tgt = pickle.load(f)

    logger.info("Tokenizing the %s data...", split)
    data_tokenized= [(sentence2tensor(lang_src, i), sentence2tensor(lang_tgt, j)) 
                        for (i, j) in zip(data_src, data_tgt)]

    logger.info("Batching %s data...", split)
    dataset = DataLoader(data_tokenized, batch_size=batch_size, num_workers=0, collate_fn=collate_data, shuffle=(True if split=="train" else False))
    return dataset, len(data_src), lang_src, lang_tgt


def train(model, train_dataset, dev_dataset, args, optimizer, lang_tgt):
    """Train the model and evaluate every epoch.
    Args:
        model: a torch.nn.Module
        train_dataset:  an iterable data loader of tokenized training dataset.
        dev_dataset: an iterable data loader of tokenized devdataset.
        args: argpase Namespace object of model and training hyperparemeters.
        optimizer: optimizer for parameters of model.
        lang_tgt: lang_tgt: DataDict object of target outputs.
    """
    
    best_val_score = 0.0
    for epoch in range(args.epoch):
        logger.info("Epoch {}/{}".format(epoch + 1, args.epoch))
        # Training the model
        model.train()
        # summary for current training loop and a running average object for loss
        sums = []
        loss_avg = {"loss": 0, "steps": 0}
        
        pbar = tqdm(train_dataset, leave=True)
        for i, (train_batch, ground_truth_batch) in enumerate(pbar):
            output_batch = model(train_batch, ground_truth_batch)
            output_batch = output_batch.reshape(-1, model.decoder.output_dim)
            ground_truth_batch = ground_truth_batch.reshape(-1)
            loss = loss_fn(output_batch, ground_truth_batch)

            # Compute gradients for all variables
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() #update with the gradients

            # detach and save summary at each 1000 steps
            if i % 1000 == 0:
                output_batch = output_batch.argmax(-1).reshape(train_batch.size(0), -1)
                ground_truth_batch = ground_truth_batch.reshape(train_batch.size(0),-1)

                output_batch = output_batch.data.cpu().numpy()
                ground_truth_batch = ground_truth_batch.data.cpu().numpy()
                pred_sens, truth_sens = pred2sentence(output_batch, ground_truth_batch, lang_tgt)

                summary_batch = {metric: metrics[metric](pred_sens, 
                                                         truth_sens) for metric in metrics}
                summary_batch["loss"] = loss.item()
                sums.append(summary_batch)
            
            loss_avg["loss"] += loss.item()
            loss_avg["steps"] += 1
            pbar.set_postfix(loss='{:05.6f}'.format(loss_avg["loss"]/float(loss_avg["steps"])))
        
        metrics_mean = {metric: np.mean([s[metric] for s in sums]) for metric in sums[0]}
        print("- Training metrics : " + str(metrics_mean))

        val_metrics = evaluate(model, loss_fn, dev_dataset, metrics, args, lang_tgt)

        val_score = val_metrics[args.metric]
        is_best = val_score > best_val_score
        # Save weights
        save_checkpoint({'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optim_dict': optimizer.state_dict()},
                        is_best=is_best,
                        checkpoint=args.save_dir,
                        )
        # Save best model path
        if is_best:
            best_val_score = val_score
            best_json_path = os.path.join(
                args.save_dir, "metrics_val_best_weights.json")
            save_dict_to_json(val_metrics, best_json_path)
        
        last_json_path = os.path.join(
            args.save_dir, "metrics_val_last_weights.json")
        save_dict_to_json(val_metrics, last_json_path)

def evaluate(model, loss_fn, dev_dataset, metrics, args, lang_tgt):
    """Evaluate the model on at the end of epoch.
    Args:
        model: a torch.nn.Module.
        loss_fn: a function that takes output_batch and ground_truth_batch and computes the loss for the batch.
        dev_dataset: an iterable data loader of tokenized devdataset.
        metrics: a dictionary of functions that compute a metric using the output and ground truth of each batch
        args: argpase Namespace object of model and training hyperparemeters.
        lang_tgt: lang_tgt: DataDict object of target outputs.
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    sums = []

    # compute metrics over the dataset
    for data in tqdm(dev_dataset, leave=True):
        # fetch the next evaluation batch
        data_batch, ground_truth_batch = data
        
        # compute model output
        output_batch = batch_pred(model, data_batch, args.max_len)
        output_batch = output_batch.reshape(-1, model.decoder.output_dim)
        ground_truth_batch = ground_truth_batch.reshape(-1)
        
        loss = loss_fn(output_batch, ground_truth_batch)

        output_batch = output_batch.argmax(-1).reshape(data_batch.size(0), -1)
        ground_truth_batch = ground_truth_batch.reshape(data_batch.size(0),-1)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        ground_truth_batch = ground_truth_batch.data.cpu().numpy()

        pred_sens, truth_sens = pred2sentence(output_batch, ground_truth_batch, lang_tgt)

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](pred_sens, truth_sens)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        sums.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([s[metric] for s in sums]) for metric in sums[0]}
    print("- Eval metrics : " + str(metrics_mean))
    return metrics_mean

def batch_pred(model, data_batch, max_len):
    '''Compute predictions our model with input tokens only
    Args:
        model: a torch.nn.Module.
        data_batch: a tensor of batched inputs.
        max_len: the maximun length of input.
    Return:
        output_batch: decoder's output of dimension [batch_size, seq_len, decoder_output_dim]
    '''
    batch_size = data_batch.size(0)
    encoder_outputs, hidden = model.encoder(data_batch)
    decoder_input = torch.repeat_interleave(torch.tensor([SOS_TOKEN]), batch_size ,dim=0).to(device) # SOS
    outputs = torch.zeros(max_len, batch_size, model.decoder.output_dim).to(device) # decoder outputs
    for t in range(1, max_len):  
        decoder_output, hidden = model.decoder(decoder_input, hidden, encoder_outputs)
        outputs[t] = decoder_output
        topi = decoder_output.argmax(1)
        decoder_input = topi
    # outputs: [max_len, batch, output_dim]
    return outputs.transpose(0,1)


def main():
    parser = argparse.ArgumentParser()
    # data and model parameters
    parser.add_argument("--train_path", type=str, default="data/train_split.txt", help="The training file path.")
    parser.add_argument("--dev_path", type=str, default="data/dev_split.txt", help="The validation file path.")
    parser.add_argument("--max_len", type=int, default=32, help="The max length of input sequence.")
    parser.add_argument("--num_layers", type=int, default=2, help="The num of layers of RNN's model.")
    parser.add_argument("--embed_dim", type=int, default=256, help="The embedding dimension.")
    parser.add_argument("--hidden_dim", type=int, default=512, help="The hidden dimension for encoder-decoder model.")
    # training parameters
    parser.add_argument("--batch_size", type=int, default=128, help="The bach size for training.")
    parser.add_argument("--dev_batch_size", type=int, default=128, help="The bach size for validation.")
    parser.add_argument("--epoch", type=int, default=10, help="Num of training epochs.")
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3, help="The learning rate for training.")
    parser.add_argument("--dropout", type=float, default=0.5, help="The dropout rate for training.")
    parser.add_argument("--metric", type=str, default="accuracy", help="Evaluation metric for training.")
    parser.add_argument("--save_dir", type=str, default="saved_models", help="The saved model path.")
    args = parser.parse_args()
    random.seed(28)

    # set random seeds
    torch.manual_seed(28)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(28)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    train_dataset, args.train_size, lang_src, lang_tgt = load_data(args.train_path, "train", args.batch_size)
    dev_dataset, args.dev_size, _, _ = load_data(args.dev_path, "dev", args.dev_batch_size)

    attention = Attention(args.hidden_dim, args.hidden_dim)

    encoder = Encoder(
        input_dim=lang_src.n_words, 
        enc_hidden_dim=args.hidden_dim, 
        dec_hidden_dim=args.hidden_dim, 
        embed_dim=args.embed_dim,
        dropout=args.dropout, 
        bidirectional=False,
    )

    decoder = Decoder(
        attention=attention,
        output_dim=lang_tgt.n_words, 
        enc_hidden_dim=args.hidden_dim, 
        dec_hidden_dim=args.hidden_dim, 
        embed_dim=args.embed_dim,
        dropout=args.dropout, 
        bidirectional=False,
    )

    model = Seq2Seq(encoder, decoder, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # initialize model weights
    model.init_weights()
    #print(model)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(model))
    train(model, train_dataset, dev_dataset, args, optimizer, lang_tgt)

    save_args(args)

if __name__ == "__main__":
    main()
