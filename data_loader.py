import random
import numpy as np
import os
import sys
import re

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3

class DataDict:
    '''
    Store sequence and split it into words.
    Modified from https://www.guru99.com/seq2seq-model.html
    '''
    def __init__(self):
        self.word2index = {}
        self.index2word = { 0: "[pad]", 1: "[sos]", 2: "[eos]", 3: "[unk]"}
        self.word2count = {}
        self.n_words = 4

    def addSentence(self,sentence):
        words = sentence2word(sentence)
        for word in words:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def sentence2word(sentence):
    '''Split sequence.'''
    return re.findall(r"sin|cos|tan|\+|\-|\*+|\(|\)|\w", sentence.strip().lower())

def sentence2idx(lang,sentence):
    '''Get index from data dictionary.'''
    lst =[]
    for word in sentence2word(sentence):
        if word in lang.word2index.keys():
            lst.append(lang.word2index[word])
        else:
            lst.append(UNK_TOKEN)
    return lst


def sentence2tensor(lang,sentence):
    '''sentence to tokens.'''
    tokens = sentence2idx(lang,sentence)
    tokens = [SOS_TOKEN] + tokens + [EOS_TOKEN]
    return tokens

def collate_data(batch):
    '''Process the list of samples to form a batch of dataloader.'''
    input_text = [torch.LongTensor(d[0]).to(device) for d in batch]
    output_text = [torch.LongTensor(d[1]).to(device) for d in batch]
    #inputs, outputs = batch
    input_text[0] = nn.ConstantPad1d((0, 32 - input_text[0].size(0)), PAD_TOKEN)(input_text[0]) 
    output_text[0] = nn.ConstantPad1d((0, 32 - output_text[0].size(0)), PAD_TOKEN)(output_text[0])
    inputs = pad_sequence(input_text, padding_value=PAD_TOKEN, batch_first=True) #batch*maxlen
    outputs = pad_sequence(output_text, padding_value=PAD_TOKEN, batch_first=True)

    return inputs, outputs
