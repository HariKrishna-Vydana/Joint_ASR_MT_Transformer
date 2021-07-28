#! /usr/bin/python

#*******************************
import sys
import os
from os.path import join, isdir
from random import shuffle
import glob




sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/MT_Transformer')

from MT_TransV1.Load_sp_model import Load_sp_models
from MT_TransV1.Make_ASR_scp_text_format_fast import format_tokenize_data

import MT_TransV1.MT_Transformer_arg
from MT_TransV1.MT_Transformer_arg import parser
args = parser.parse_args()


if not isdir(args.data_dir):
        os.makedirs(args.data_dir)

#name_tag=str(sys.argv[1])

format_tokenize_data(scp_files=glob.glob(args.train_path + "*"),transcript=args.src_text_file,Translation=args.tgt_text_file,outfile=open(join(args.data_dir,'train_scp'),'a'),Src_model_path=args.Src_model_path,Tgt_model_path=args.Tgt_model_path)
format_tokenize_data(scp_files=glob.glob(args.dev_path + "*"),transcript=args.src_text_file,Translation=args.tgt_text_file,outfile=open(join(args.data_dir,'dev_scp'),'a'), Src_model_path=args.Src_model_path,Tgt_model_path=args.Tgt_model_path)



