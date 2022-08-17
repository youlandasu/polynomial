import os
import numpy as np
import copy
import math
import random
from typing import Optional, Tuple
from dataclasses import dataclass, field
from itertools import chain
from data_loader import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''Modified from https://www.guru99.com/seq2seq-model.html for supporting batched GPU.'''
class Encoder(nn.Module):
    def __init__(
        self, 
        input_dim, 
        enc_hidden_dim=512, 
        dec_hidden_dim=512, 
        embed_dim=256, 
        dropout=0.5,
        bidirectional=False):
        super(Encoder, self).__init__()
        
        #set the encoder input dimesion , embed dimesion, hidden dimesion, and number of layers 
        self.embed_dim = embed_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim

        #initialize the embedding layer with input and embed dimention
        self.embedding = nn.Embedding(input_dim, self.embed_dim)
        #TODO: pos embedding
        #intialize the GRU to take the input dimetion of embed, and output dimention of hidden and
        #set the number of gru layers
        self.rnn = nn.GRU(self.embed_dim, self.enc_hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.dire = 2 if bidirectional else 1
        self.linear = nn.Linear(self.enc_hidden_dim*self.dire, dec_hidden_dim)
        self.dropout = nn.Dropout(dropout)
                
    def forward(self, src):
        
        embedded = self.dropout(self.embedding(src))
        # outputs: [batch, len, hid_dim*2]
        # hidden: [layers*2, batch, hid_dim]
        outputs, hidden = self.rnn(embedded)
        if self.dire==2:
            hidden = torch.tanh(self.linear(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        else:
            hidden = torch.tanh(self.linear(hidden[-1,:,:]))
        #hidden = [batch size, dec hid dim]
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(
        self, 
        attention,
        output_dim, 
        enc_hidden_dim=512, 
        dec_hidden_dim=512, 
        embed_dim=256, 
        dropout=0.5,
        bidirectional=False,
        ):
        super(Decoder, self).__init__()

        #set the encoder output dimension, embed dimension, hidden dimension, and number of layers 
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.attention = attention

        # initialize every layer with the appropriate dimension. For the decoder layer, it will consist of an embedding, GRU, a Linear layer and a Log softmax activation function.
        self.embedding = nn.Embedding(output_dim, self.embed_dim)
        self.dire = 2 if bidirectional else 1
        self.rnn = nn.GRU(self.embed_dim + self.enc_hidden_dim * self.dire, self.dec_hidden_dim, batch_first=True)
        self.linear = nn.Linear(self.embed_dim + self.enc_hidden_dim * self.dire + self.dec_hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):

        
        input = input.unsqueeze(1)#input: [batch,1]
        embedded = self.dropout(self.embedding(input)) #[batch,1,embed_dim]
        atts = self.attention(hidden, encoder_outputs) #[batch,seq_len]
        att_encoder_outputs = torch.matmul(atts.unsqueeze(1), encoder_outputs) #[batch,1, enc_hidden_dim*2]
        decoder_inputs = torch.cat((att_encoder_outputs,embedded), dim=-1) #[batch,1, enc_hidden_dim*2+embed_dim]
        
        output, hidden = self.rnn(decoder_inputs, hidden.unsqueeze(0))
        # outputs: [batch,1,dec_hidden_dim]
        # hidden: [1, batch, hid_dim]
        assert (output.view(-1, self.dec_hidden_dim) == hidden.view(-1, self.dec_hidden_dim)).all()

        prediction = self.linear(torch.cat((output.squeeze(1), 
                                            att_encoder_outputs.squeeze(1),
                                            embedded.squeeze(1)), dim=-1))

        # prediction: [batch, output_dim]
        return prediction, hidden.squeeze(0)

class Attention(nn.Module):
    '''Copyed from https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb'''
    def __init__(self, enc_hid_dim, dec_hid_dim, bidirectional=False):
        super().__init__()
        
        self.dire = 2 if bidirectional else 1
        self.attn = nn.Linear((enc_hid_dim * self.dire) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        src_len = encoder_outputs.shape[1]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        #attention= [batch size, src len]    

        return F.softmax(attention, dim=1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, max_len=32):
        super().__init__()
      
        #initialize the encoder and decoder
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
     
    def forward(self, source, target, teacher_forcing_ratio=0.5):

        input_length = source.size(1) #get the input length (number of words in sentence)
        batch_size = target.size(0)
        target_length = target.size(1)
        vocab_size = self.decoder.output_dim
        
        #initialize a variable to hold the predicted outputs
        outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)

        #encode every word in a sentence
        encoder_outputs, hidden = self.encoder(source)
    
        #add a token before the first predicted word
        decoder_input = torch.repeat_interleave(torch.tensor([SOS_TOKEN]),batch_size,dim=0).to(device) # SOS
        #topk is used to get the top K value over a list

        for t in range(1, target_length):  
            decoder_output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs[t] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            topi = decoder_output.argmax(1)
            decoder_input = (target[:, t] if teacher_force else topi)

        return outputs.transpose(0,1) #[batch, seq_len, vocab_size]

    def init_weights(self):
        def helper(model):
            for name, parameters in model.named_parameters():
                if "weight" in name:
                    nn.init.normal_(parameters.data, mean=0, std=0.01)
                else:
                    nn.init.constant_(parameters.data, 0)
        return self.apply(helper)

def loss_fn(outputs, labels):
    """Compute the cross entropy loss given decoder's outputs and ground truth target tokens.
    Args:
        outputs: dimension batch_size*seq_len x target_vocab_size 
        labels: dimension batch_size x seq_len where each element is a label in [0, 1, ... target_vocab_size-1].
    Return:
        loss: cross entropy loss in the batch
    """

    # cross entropy loss
    loss_fct = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    loss = loss_fct(outputs, labels)

    return loss


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the predicted sentences and ground truth sentences.
    Args:
        outputs: list of strings and each string is a predicted sentence
        labels: list of strings and each string is a ground truth sentence
    Return: (float) accuracy in [0,1]
    """
    acc = [int(output==label) for output, label in zip(outputs, labels)]
    return np.mean(acc)

# maintain all metrics in each training and evaluation loop
metrics = {
    'accuracy': accuracy,
}