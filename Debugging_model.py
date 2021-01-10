#!/usr/bin/python
import sys
import os
import subprocess
from os.path import join, isdir
import numpy as np
import fileinput
import json
import random
from itertools import chain
from numpy.random import permutation
##------------------------------------------------------------------
import torch
from torch.autograd import Variable
#----------------------------------------
import torch.nn as nn
from torch import autograd, nn, optim
os.environ['PYTHONUNBUFFERED'] = '0'
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim

from random import shuffle
from statistics import mean
import matplotlib
import matplotlib.pyplot as plt 
plt.switch_backend('agg')
matplotlib.pyplot.viridis()
import glob
import pickle

#*************************************************************************************************************************
####### Loading the Parser and default arguments
sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/Joint_ASR_MT_Transformer/ASR_MT_Transv1')
import ASR_MT_Transformer_arg
from ASR_MT_Transformer_arg import parser
args = parser.parse_args()

from Set_gpus import Set_gpu
if args.gpu:
    Set_gpu()

# Src_model=Load_sp_models(args.Src_model_path)
# Tgt_model=Load_sp_models(args.Tgt_model_path)
# ###initilize the model
# model,optimizer=Initialize_Att_model(args)
# #============================================================
# #------------------------------------------------------------  
# train_gen = DataLoader(files=glob.glob(args.data_dir + "train_scp"),
#                         max_batch_label_len=args.max_batch_label_len,
#                         max_batch_len=args.max_batch_len,
#                         max_feat_len=args.max_feat_len,
#                         max_label_len=args.max_label_len,
#                         Src_model=Src_model,
#                         Tgt_model=Tgt_model)
sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/Joint_ASR_MT_Transformer/ASR_MT_Transv1')
#==============================================================
#from Dataloader_for_MT_v2 import DataLoader
from TRANSFORMER_ASR_MT_V1 import Transformer
from Initializing_Transformer_ASR_MT import Initialize_Att_model
#from Training_loop_ASR_MT import train_val_model
#from Load_sp_model import Load_sp_models
#==============================================================


Src_model_path='/mnt/matylda3/vydana/HOW2_EXP/Joint_ASR_MT_Transformer/Tokenizers/models_ENG/ENG_Tok__bpe.model'
Tgt_model_path='/mnt/matylda3/vydana/HOW2_EXP/Joint_ASR_MT_Transformer/Tokenizers/models_PTG/PTG_Tok__bpe.model'

src_text_file='/mnt/matylda3/vydana/HOW2_EXP/Joint_ASR_MT_Transformer/Tokenizers/full_text_id.en_normalized'
tgt_text_file='/mnt/matylda3/vydana/HOW2_EXP/Joint_ASR_MT_Transformer/Tokenizers/full_text_id.pt_normalized'

a_file = open("data.pkl", "rb")
data_dict=pickle.load(a_file)
print(data_dict.keys())


args.src_text_file=src_text_file
args.tgt_text_file=tgt_text_file
args.Src_model_path=Src_model_path
args.Tgt_model_path=Tgt_model_path

model,optimizer=Initialize_Att_model(args)

#print(model)
B1=data_dict
smp_Src_data = B1.get('smp_Src_data')
smp_Src_labels = B1.get('smp_Src_labels')
smp_Tgt_labels = B1.get('smp_Tgt_labels')


###---------------------------###
input = torch.from_numpy(smp_Src_data).float()

#input = torch.from_numpy(smp_Src_data).float() ####
smp_Src_labels = torch.LongTensor(smp_Src_labels)
smp_Tgt_labels = torch.LongTensor(smp_Tgt_labels)
Decoder_out_dict = model(input,smp_Src_labels,smp_Tgt_labels)

#print(model)

cost=Decoder_out_dict.get('cost')
cost.backward()
optimizer.step()
#print(Decoder_out_dict.keys())
