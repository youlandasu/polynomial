import random
from argparse import Namespace
import os
import logging
import pickle
import json
from tqdm import tqdm

import sys
import numpy as np
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, DataDict, sentence2tensor
from model import Seq2Seq, Encoder, Decoder, Attention, loss_fn, metrics
from utils import load_checkpoint, find_end

MAX_SEQUENCE_LENGTH = 29
TRAIN_URL = "https://scale-static-assets.s3-us-west-2.amazonaws.com/ml-interview/expand/train.txt"

logger = logging.getLogger(__name__)
torch.cuda.empty_cache() 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions


def score(true_expansion: str, pred_expansion: str) -> int:
    """ the scoring function - this is how the model will be evaluated

    :param true_expansion: group truth string
    :param pred_expansion: predicted string
    :return:
    """
    return int(true_expansion == pred_expansion)


# --------- START OF IMPLEMENT THIS --------- #
def predict(factors: str):
    saved_path = "saved_models"
    config_file = "config.json"
    # Load model configerations
    with open(os.path.join(saved_path, config_file),"r") as fin:
        loaded_args = json.load(fin)
    args = Namespace(**loaded_args)

    # load data dictionary
    with open("lang_src.pickle","rb") as f:
        lang_src = pickle.load(f)
    with open("lang_tgt.pickle","rb") as f:
        lang_tgt = pickle.load(f)

    # config model
    attention = Attention(args.hidden_dim, args.hidden_dim, bidirectional=False)

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
    # Reload weights from the saved file
    load_checkpoint(os.path.join(saved_path, 'best.pth.tar'), model)

    # sentence to tensor
    tokens = sentence2tensor(lang_src, factors)
    input_text = torch.LongTensor(tokens).unsqueeze(0).to(device)
    input_text = nn.ConstantPad1d((0, 32 - input_text.size(1)), PAD_TOKEN)(input_text) #[batch=1, len=32]
    # start prediction
    model.eval()
    encoder_outputs, hidden = model.encoder(input_text)
    decoder_input = torch.tensor([SOS_TOKEN]).to(device)
    outputs = torch.zeros(args.max_len, 1, model.decoder.output_dim).to(device) # decoder outputs
    for t in range(1, args.max_len):  
        decoder_output, hidden = model.decoder(decoder_input, hidden, encoder_outputs)
        outputs[t] = decoder_output
        topi = decoder_output.argmax(1)
        decoder_input = topi
    outputs = outputs.transpose(0,1).reshape(-1, model.decoder.output_dim)
    outputs = outputs.argmax(-1)
    outputs = outputs.data.cpu().numpy()
    endi = find_end(outputs)
    words_idx = outputs[1:endi]
    words = [lang_tgt.index2word[id] for id in words_idx]
    return "".join(words)


# --------- END OF IMPLEMENT THIS --------- #


def main(filepath: str):
    # set random seeds
    torch.manual_seed(23)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(23)

    factors, expansions = load_file(filepath)
    pred = [predict(f) for f in tqdm(factors, desc="Predicting test examples...", leave=True)]
    scores = [score(te, pe) for te, pe in zip(expansions, pred)]
    print(np.mean(scores))


if __name__ == "__main__":
    #main("test.txt" if "-t" in sys.argv else "train.txt")
    main("test_split.txt")