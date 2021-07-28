#!/usr/bin/python
import sys
import os
import subprocess

from os.path import join, isdir, isfile
import torch
import json

import numpy as np
from torch import autograd, nn, optim
import torch.nn.functional as F

from argparse import Namespace
#**********

def Load_Ens_ASR_model(pre_trained_weight):
        #=================================================================
        #**********
        #Loading the Parser and default arguments
        #sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/MT_Transformer/MT_TransV1/')

        sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/ASR_Transformer')
        import ASR_TransV1.Transformer_arg
        from ASR_TransV1.Transformer_arg import parser
        from ASR_TransV1.Initializing_Transformer_ASR import Initialize_Att_model as Initialize_Trans_model
        #===================================================================
        
        model_dir="/".join(pre_trained_weight.split('/')[:-1]) 
        model_path_name=join(model_dir,'model_architecture_')
        
        ###load the architecture if you have to load
        with open(model_path_name, 'r') as f:
                TEMP_args = json.load(f)

        args = Namespace(**TEMP_args)
        args.gpu=0
        args.pre_trained_weight=pre_trained_weight
        model,optimizer=Initialize_Trans_model(args)
        model.eval()
        return model,optimizer


