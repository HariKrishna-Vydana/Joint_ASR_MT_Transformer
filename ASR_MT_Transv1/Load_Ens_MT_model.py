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

sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/MT_Transformer/')
import MT_TransV1.MT_Transformer_arg
from MT_TransV1.MT_Transformer_arg import parser
from MT_TransV1.Initializing_Transformer_MT import Initialize_Att_model as Initialize_Trans_model


def Load_Ens_MT_model(pre_trained_weight):
        #=================================================================
        #**********
        #breakpoint()
        #Loading the Parser and default arguments
        #sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/MT_Transformer/MT_TransV1/')
        #import MT_Transformer_arg
        #from MT_Transformer_arg import parser
        #from Initializing_Transformer_MT import Initialize_Att_model as Initialize_Trans_model
        #===================================================================

        model_dir="/".join(pre_trained_weight.split('/')[:-1]) 
        model_path_name=join(model_dir,'model_architecture_')
        
        ###load the architecture if you have to load
        with open(model_path_name, 'r') as f:
                TEMP_args = json.load(f)

        args = Namespace(**TEMP_args)
        #print(args)
        args.gpu=0
        args.pre_trained_weight=pre_trained_weight
        model,optimizer=Initialize_Trans_model(args)
        model.eval()

        return model,optimizer


