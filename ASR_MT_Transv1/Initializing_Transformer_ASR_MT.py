#!/usr/bin/python
import sys
import os
from os.path import join, isdir
import torch
from torch import optim


sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/Joint_ASR_MT_Transformer/ASR_MT_Transv1')
from TRANSFORMER_ASR_MT_V1 import Transformer,TransformerOptimizer
from utils__ import count_parameters
#====================================================================================
def Initialize_Att_model(args):
        #breakpoint()
        model = Transformer(args)
        model= model.cuda() if args.gpu else model

        trainable_parameters=list(model.parameters())    
        
        ###optimizer
        optimizer = optim.Adam(params=trainable_parameters,lr=args.learning_rate,betas=(0.9, 0.99))

        #optimizer with warmup_steps
        optimizer=TransformerOptimizer(optimizer=optimizer,k=args.lr_scale, d_model=args.decoder_dmodel, step_num=args.step_num, warmup_steps=args.warmup_steps)

        pre_trained_weight=args.pre_trained_weight
        weight_flag=pre_trained_weight.split('/')[-1]
        print("Initial Weights",weight_flag)

        if weight_flag != '0':
                print("Loading the model with the weights form:",pre_trained_weight)
                weight_file=pre_trained_weight.split('/')[-1]
                weight_path="/".join(pre_trained_weight.split('/')[:-1])
                enc_weight=join(weight_path,weight_file)
                model.load_state_dict(torch.load(enc_weight, map_location=lambda storage, loc: storage),strict=True)                

                optimizer_name=join(weight_path,weight_file,'_opt')
                if os.path.isfile(optimizer_name):
                        optimizer.load_state_dict(torch.load(optimizer_name, map_location=lambda storage, loc: storage))

        print("model:=====>",(count_parameters(model))/1000000.0)
        return model, optimizer
#====================================================================================
#====================================================================================
