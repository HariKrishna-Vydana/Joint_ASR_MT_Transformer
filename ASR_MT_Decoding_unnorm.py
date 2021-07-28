#!/usr/bin/python
import sys
import os
from os.path import join, isdir, isfile
#----------------------------------------
import glob
import json
from argparse import Namespace

import torch

#**********
sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/Joint_ASR_MT_Transformer')
#from Initializing_model_LSTM_SS_v2_args import Initialize_Att_model
from ASR_MT_Transv1.Load_sp_model import Load_sp_models
from ASR_MT_Transv1.utils__ import plotting,read_as_list
from ASR_MT_Transv1.user_defined_losses import compute_cer
from ASR_MT_Transv1.Decoding_loop import get_Bleu_for_beam
from ASR_MT_Transv1.Load_Encode_sp_model import Load_Encode_sp_model
from ASR_MT_Transv1.get_best_weights import get_best_weights

from ASR_MT_Transv1.Initializing_Transformer_ASR_MT_unnorm import Initialize_Att_model
#-----------------------------------

import ASR_MT_Transv1.ASR_MT_Transformer_arg
from ASR_MT_Transv1.ASR_MT_Transformer_arg import parser
args = parser.parse_args()

model_path_name=join(args.model_dir,'model_architecture_')
print(model_path_name)
#
#
#--------------------------------
###load the architecture if you have to load
with open(model_path_name, 'r') as f:
        TEMP_args = json.load(f)

ns = Namespace(**TEMP_args)
args=parser.parse_args(namespace=ns)

#---    Load the models for ensembling
#MT_model
from ASR_MT_Transv1.Load_Ens_MT_model import Load_Ens_MT_model 
Ens_MT_model,opt__=Load_Ens_MT_model(args.pre_trained_MT_weight)
args.MT_model=Ens_MT_model



from ASR_MT_Transv1.Load_Ens_ASR_model import Load_Ens_ASR_model
Ens_ASR_model,opt__=Load_Ens_ASR_model(args.pre_trained_ASR_weight)
args.ASR_model=Ens_ASR_model




#exit(0)
if not isdir(args.model_dir):
        os.makedirs(args.model_dir)
args.Am_weight = 1
args.LM_model = None
##==================================
##**********************************
##**********************************
def main():
        #Load the model from architecture
        model,optimizer=Initialize_Att_model(args)
        model.eval()
        args.gpu=False
        
        ###make SWA name 
        model_name = str(args.model_dir).split('/')[-1]
        ct=model_name+'_SWA_random_tag_'+str(args.SWA_random_tag)
        


        ##check the Weight averaged file and if the file does not exist then lcreate them
        ## if the file exists load them
        #### SWA is done seperately with Get_SWA.sh and Get_SWA_weights.py
        ####
        ###
        #if not isfile(join(args.model_dir,ct)):
        #    model_names,checkpoint_ter = get_best_weights(args.weight_text_file,args.Res_text_file)
        #    model_names_checkpoints=model_names[:args.early_stopping_checkpoints]
        #    model = Stocasting_Weight_Addition(model,model_names_checkpoints)
        #    torch.save(model.state_dict(),join(args.model_dir,ct))
        #else:
        #    print("taking the weights from",ct,join(args.model_dir,str(ct)))
        #    args.pre_trained_weight = join(args.model_dir,str(ct))
        #    model,optimizer=Initialize_Att_model(args)
        #---------------------------------------------
        #model.eval() 
        #print("best_weight_file_after stocastic weight averaging")
        #---------------------------------------------
        #=================================================
        model = model.cuda() if args.gpu else model
        plot_path=join(args.model_dir,'decoding_files','plots')
        #=================================================
        #=================================================
        ####read all the scps and make large scp with each lines as a feature
        decoding_files_list=glob.glob(args.dev_path + "*")
        scp_paths_decoding=[]
        for i_scp in decoding_files_list:
            scp_paths_decoding_temp=open(i_scp,'r').readlines()
            scp_paths_decoding+=scp_paths_decoding_temp

        #scp_paths_decoding this should contain all the scp files for decoding
        #====================================================
        ###sometime i tend to specify more jobs than maximum number of lines in that case python indexing error we get  
        job_no=int(args.Decoding_job_no)-1
        
        #args.gamma=0.5
        #print(job_no)
            
        #####get_cer_for_beam takes a list as input
        present_path = scp_paths_decoding[job_no]
        
        Src_text_file_dict = {line.split(' ')[0]:" ".join(line.strip().split(' ')[1:]) for line in open(args.src_text_file)}
        Tgt_text_file_dict = {line.split(' ')[0]:" ".join(line.strip().split(' ')[1:]) for line in open(args.tgt_text_file)}
        
        key = present_path.split(' ')[0]
        Src_text = Src_text_file_dict.get(key,None)        
        Tgt_text = Tgt_text_file_dict.get(key,None)          
        
        Src_text = Src_text.strip() if Src_text else Src_text
        Tgt_text = Tgt_text.strip() if Tgt_text else Tgt_text

        scp_paths_decoding = present_path.split(' ')[1]

        
        if not Src_text:
            print("utterance not present in source tokens something wrong",key)
            exit(0)
        else:
            Src_tokens = Load_Encode_sp_model(args.Src_model_path,Src_text)
            Tgt_tokens = Load_Encode_sp_model(args.Tgt_model_path,Tgt_text)
        
        #print(scp_paths_decoding, key, Src_tokens, Src_text, Tgt_tokens, Tgt_text, model, plot_path)
        get_Bleu_for_beam(scp_paths_decoding, key, Src_tokens, Src_text, Tgt_tokens, Tgt_text, model, plot_path, args)

#--------------------------------
#--------------------------------

if __name__ == '__main__':
    main()

