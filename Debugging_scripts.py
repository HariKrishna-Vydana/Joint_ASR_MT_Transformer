#!/usr/bin/python
import sys
import os
import subprocess
from os.path import join, isdir
import numpy as np
import fileinput
from numpy.random import permutation

import torch
from torch import autograd, nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
#----------------------------------------
from random import shuffle
import glob
from statistics import mean
import json
import kaldi_io


import pickle


# scp_file='/mnt/matylda3/vydana/benchmarking_datasets/Timit/scp_files/train/sorted_feats_pdnn_train_scp'
# transcript='/mnt/matylda3/vydana/benchmarking_datasets/Timit/All_text'
# Translation='/mnt/matylda3/vydana/benchmarking_datasets/Timit/All_text'

#Word_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/models/Timit_PHSEQ_100/Timit_PHSEQ__100__word.model'
#Char_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/models/Timit_PHSEQ_100/Timit_PHSEQ__100__word.model'


Word_model_path='/mnt/matylda3/vydana/HOW2_EXP/Joint_ASR_MT_Transformer/Tokenizers/models_ENG/ENG_Tok__bpe.model'
Char_model_path='/mnt/matylda3/vydana/HOW2_EXP/Joint_ASR_MT_Transformer/Tokenizers/models_PTG/PTG_Tok__bpe.model'

#src_text_file='/mnt/matylda3/vydana/HOW2_EXP/Joint_ASR_MT_Transformer/Tokenizers/full_text_id.en'
#tgt_text_file='/mnt/matylda3/vydana/HOW2_EXP/Joint_ASR_MT_Transformer/Tokenizers/full_text_id.pt'

###they contain utterance lists similar to the scp files
#train_path='/mnt/matylda3/vydana/HOW2_EXP/Joint_ASR_MT_Transformer/scp_files/train/'
#dev_path='/mnt/matylda3/vydana/HOW2_EXP/Joint_ASR_MT_Transformer/scp_files/dev/'
#test_path='/mnt/matylda3/vydana/HOW2_EXP/Joint_ASR_MT_Transformer/scp_files/dev/'



sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/Joint_ASR_MT_Transformer/ASR_MT_Transv1/')
from Load_sp_model import Load_sp_models
from Make_ASR_scp_text_format import format_tokenize_data

#***************************************************************
#from Dataloader_for_AM_v2 import DataLoader

from Dataloader_for_MT_v2 import DataLoader

#data_dir='/mnt/matylda3/vydana/HOW2_EXP/Librispeech_V2/LIBRISP960hrs_training_Data_249_scps/'
#data_dir='/mnt/matylda3/vydana/HOW2_EXP/WSJ2/WSJ80hrs_training_Data_249_scps_nocmvn/'
#data_dir="/mnt/matylda3/vydana/HOW2_EXP/Librispeech_V2/train_100_tokenization/LIBRISP100hrs_training_Data_249_scps_Nocmvn_bpe/"
#Src_model, Tgt_model
data_dir="/mnt/matylda3/vydana/HOW2_EXP/Joint_ASR_MT_Transformer/ASR_MT_data_files/"

Word_model = Load_sp_models(Word_model_path)
Char_model=Load_sp_models(Char_model_path)
train_gen = DataLoader(files=glob.glob(data_dir + "train_scp"),
                        max_batch_label_len=8000,
                        max_batch_len=100,
                        max_feat_len=2000,
                        max_label_len=800,
                        Src_model=Word_model,
                        Tgt_model=Char_model)   


exmp=0
#a_file = open("data.pkl", "wb")

for i in range(1,3000):
        B1 = train_gen.next()
        
        #for name in B1.get('smp_names'):
        exmp += len(B1.get('smp_names'))
        print(i,B1.get('smp_Src_data').shape,B1.get('smp_Src_labels').shape, B1.get('smp_Tgt_labels').shape,exmp)
        #print(i,B1.get('smp_Src_Text'), B1.get('smp_Tgt_Text'),exmp)
        #pickle. dump(B1, a_file)
        #exit(0)