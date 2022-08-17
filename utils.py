import json
import logging
import os
import shutil
import collections
import math
import numpy as np

import torch
import torch.nn as nn
from data_loader import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN

def save_args(args):
    '''Save model config to json.
    Args:
        args: argpase Namespace object
    '''
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "config.json"),"w") as fout:
        json.dump(vars(args), fout, indent=4)

def save_checkpoint(state, is_best, checkpoint):
    """Modified from https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/nlp/utils.py
    Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    os.makedirs(checkpoint, exist_ok=True)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Modified from https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/nlp/utils.py
    Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

def save_dict_to_json(d, json_path):
    """Saves training and dev metrics to json file
    Args:
        d: dictionary of metrics
        json_path: the path to json file
    """
    with open(json_path, 'w') as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

def find_end(arr):
    '''Find the EOS token of predictions.'''
    eosi_arr = np.where(arr == EOS_TOKEN)[0]
    if len(eosi_arr) > 0:
        return eosi_arr[0]
    else:
        return len(arr)

def pred2sentence(output_batch, ground_truth_batch, lang_tgt):
    '''Convert predictions to sentences.
    '''
    truth_sens = []
    pred_sens = []
    for pred, truth in zip(output_batch, ground_truth_batch):
        truth_endi = np.where(truth == EOS_TOKEN)[0][0]
        truth_words_idx = truth[1:truth_endi]
        pred_endi = find_end(pred)
        pred_words_idx = pred[1:pred_endi]
        truth_words = [lang_tgt.index2word[id] for id in truth_words_idx]
        pred_words = [lang_tgt.index2word[id] for id in pred_words_idx]
        truth_sens.append("".join(truth_words))
        pred_sens.append("".join(pred_words))
    return truth_sens, pred_sens