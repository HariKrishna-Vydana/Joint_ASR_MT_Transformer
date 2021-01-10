#! /usr/bin/python


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb 
from torch.autograd import Variable


import sys
sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/Joint_ASR_MT_Transformer/ASR_MT_Transv1')
from Trans_conv_layers import Conv_2D_Layers
from Trans_utilities import get_attn_key_pad_mask, get_subsequent_mask, get_attn_pad_mask_encoder, get_attn_pad_mask,get_encoder_non_pad_mask, get_decoder_non_pad_mask
from Trans_MHA import MultiHeadAttention, PositionwiseFeedForward, PositionalEncoding
from Load_sp_model import Load_sp_models

#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    """Compose with two sub-layers.
        1. A multi-head self-attention mechanism
        2. A simple, position-wise fully connected feed-forward network.
    """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1,ff_dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=ff_dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        x=enc_input
        nx=self.norm1(x)        
        enc_output, enc_slf_attn = self.slf_attn(nx, nx, nx, mask=slf_attn_mask)
        x=x+self.dropout(enc_output)        

        nx=self.norm2(x)
        enc_output = x+self.dropout(self.pos_ffn(nx))
        return enc_output, enc_slf_attn
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
class Encoder(nn.Module):
    """Encoder of Transformer including self-attention and feed forward. """

    def __init__(self, args, MT_flag):
        super(Encoder, self).__init__()

        ###use of MT_flag in encoder

        self.MT_flag = MT_flag
        #This always takes source model as input  embeding oth for speech or text input
        self.Src_model_path = args.Src_model_path if self.MT_flag else args.Src_model_path
        
        # parameters
        ## it can get input from conv layers or encoder dmodel 
        #so always the dimensions is always the d_model of speech 
        self.d_input = args.encoder_dmodel
        self.n_layers = args.encoder_layers_MT if self.MT_flag else args.encoder_layers
        self.n_head = args.encoder_heads_MT if self.MT_flag else args.encoder_heads
        self.d_model = args.encoder_dmodel_MT if self.MT_flag else args.encoder_dmodel
        self.d_inner = args.encoder_dinner_MT if self.MT_flag else args.encoder_dinner
        self.dropout_rate = args.encoder_dropout_MT if self.MT_flag else args.encoder_dropout
        self.encoder_ff_dropout = args.encoder_ff_dropout_MT if self.MT_flag else args.encoder_ff_dropout
        self.pe_max_len = args.pe_max_len_MT if self.MT_flag else args.pe_max_len

        self.xscale = math.sqrt(self.d_model)
        self.d_k = int(self.d_model/self.n_head) 
        self.d_v = int(self.d_model/self.n_head)
        #=======================================================
        # use linear transformation with layer norm to replace input embedding
        ###switches between ASR and MT modules 
        if self.MT_flag:
            self.linear_in = nn.Linear(self.d_input, self.d_model)
        else:
            self.linear_in = nn.Linear(self.d_input, self.d_model)
        #========================================================
        
        self.layer_norm_in = nn.LayerNorm(self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, self.dropout_rate,max_len=self.pe_max_len)
        self.layer_stack = nn.ModuleList([EncoderLayer(self.d_model, self.d_inner, self.n_head, self.d_k, self.d_v, dropout=self.dropout_rate, ff_dropout=self.encoder_ff_dropout) for _ in range(self.n_layers)])

    def forward(self, padded_input, return_attns=False):
        """ Args: padded_input: N x Ti x D  input_lengths: N Returns: enc_output: N x Ti x H """ 
        #breakpoint()
        enc_slf_attn_list = []      
        # Prepare masks

        non_pad_mask=None;
        dec_enc_attn_mask=None;
        slf_attn_mask=None;
        slf_attn_mask_keypad=None

        padded_input_norm=self.layer_norm_in(self.linear_in(padded_input))
        enc_output=self.positional_encoding(padded_input_norm)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output,non_pad_mask=non_pad_mask,slf_attn_mask=slf_attn_mask)
            
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list

        return enc_output,
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
