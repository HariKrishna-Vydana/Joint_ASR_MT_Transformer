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

import time


#*************************************************************************************************************************
####### Loading the Parser and default arguments
sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/Joint_ASR_MT_Transformer/ASR_MT_Transv1')
import ASR_MT_Transformer_arg
from ASR_MT_Transformer_arg import parser
args = parser.parse_args()

###save architecture for decoding
model_path_name=join(args.model_dir,'model_architecture_')
with open(model_path_name, 'w') as f:
    json.dump(args.__dict__, f, indent=2)
print(args)
# #####setting the gpus in the gpu cluster
# #**********************************
from Set_gpus import Set_gpu
if args.gpu:
    Set_gpu()
###----------------------------------------
#==============================================================
from Dataloader_for_MT_v4 import DataLoader
from TRANSFORMER_ASR_MT_V1 import Transformer
from Initializing_Transformer_ASR_MT import Initialize_Att_model
from Training_loop_ASR_MT import train_val_model
from Load_sp_model import Load_sp_models
#==============================================================
############################################
#=============================================================
def main():
        ##Load setpiece models for Dataloaders
        Src_model=Load_sp_models(args.Src_model_path)
        Tgt_model=Load_sp_models(args.Tgt_model_path)
        ###initilize the model
        start = time.perf_counter()        
        #============================================================#train_splits $dev_splits
        #------------------------------------------------------------ 
        train_gen = DataLoader(files=glob.glob(args.data_dir + "train_splits/*"),
                                max_batch_label_len=args.max_batch_label_len,
                                max_batch_len=args.max_batch_len,
                                max_feat_len=args.max_feat_len,
                                max_label_len=args.max_label_len,
                                Src_model=Src_model,
                                Tgt_model=Tgt_model)    

        #Flags that may change while training
        print('before for loop') 
        for i in range(1000):
            B1 = train_gen.next()
            assert B1 is not None, "None should never come out of the DataLoader"
            #print(B1.keys())
            smp_Src_data = B1.get('smp_Src_data')

            time.sleep(0.1)
            #breakpoint()
            smp_Src_labels = B1.get('smp_Src_labels')
            smp_Tgt_labels = B1.get('smp_Tgt_labels') 
            ###for future
            smp_Src_Text = B1.get('smp_Src_Text') 
            smp_Tgt_Text = B1.get('smp_Tgt_Text')
            

            smp_Src_labels_tc = B1.get('smp_Src_labels_tc')
            smp_Src_Text_tc = B1.get('smp_Src_Text_tc')
 
            smp_src_data = torch.from_numpy(smp_Src_data).float()
            MT_utterances = torch.sum(torch.sum(smp_src_data,dim=1,keepdim=True),dim=2,keepdim=True)==0
            #print('i :====>',i,smp_Src_data,smp_Src_labels,smp_Tgt_labels,smp_Src_Text,smp_Tgt_Text,smp_Src_labels_tc,smp_Src_Text_tc)
            print('i :====>',i,smp_Src_data.shape)    
            # print('i :====>',i, "smp_Src_data,smp_Src_labels,smp_Tgt_labels,smp_Src_Text,smp_Tgt_Text",smp_Src_data.shape,smp_Src_labels.shape,smp_Tgt_labels.shape,smp_Src_Text.shape,smp_Tgt_Text.shape)#,smp_Src_data[0],smp_Src_labels[0],smp_Tgt_labels[0],smp_Src_Text[0],smp_Tgt_Text[0])
        #======================================


        finish=time.perf_counter()
        print(f'Finished in {round(finish-start,2)} secound(s)')
        exit(0)
#=======================================================
#=============================================================================================
if __name__ == '__main__':
    main()



