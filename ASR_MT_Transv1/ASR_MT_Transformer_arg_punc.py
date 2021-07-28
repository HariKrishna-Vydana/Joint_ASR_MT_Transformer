#!/usr/bin/pyhton
import sys
import argparse


parser = argparse.ArgumentParser(description="")
parser.add_argument("--gpu",metavar='',type=int,default='0',help="use gpu flag 0|1")
#---------------------------


parser.add_argument("--apply_cmvn",metavar='',type=int,default='1',help="apply_cmvn")

###convolution parameters
parser.add_argument("--hidden_size",metavar='',type=int,default='320',help="Space token for the Char model")
parser.add_argument("--input_size",metavar='',type=int,default='249',help="Space token for the Char model")
parser.add_argument("--kernel_size",metavar='',type=int,default='3',help="kernel_size")
parser.add_argument("--stride",metavar='',type=int,default='2',help="stride")
parser.add_argument("--in_channels",metavar='',type=int,default='1',help="in_channels")
parser.add_argument("--out_channels",metavar='',type=int,default='256',help="out_channels")
parser.add_argument("--conv_dropout",metavar='',type=float,default='0.1',help="conv_dropout")
parser.add_argument("--isresidual",metavar='',type=int,default=0,help="isresidual --> 1|0 ") ###removed 
parser.add_argument("--enc_front_end",metavar='',type=str,default='Subsamp_lstm',help="Subsamp_lstm|conv2d|nothing") ###remove
parser.add_argument("--Conv_Act",metavar='',type=str,default='relu',help="relu|tanh") ##remove



###ASR_parameters
parser.add_argument("--encoder_dropout",metavar='',type=float,default='0.3',help="encoder dropout ")
parser.add_argument("--encoder_layers",metavar='',type=int,default='4',help="encoder dropout ")
parser.add_argument("--encoder_dmodel",metavar='',type=int,default='256',help="encoder_dmodel")
parser.add_argument("--encoder_heads",metavar='',type=int,default='4',help="encoder_heads")
parser.add_argument("--encoder_dinner",metavar='',type=int,default='1024',help="encoder_dinner")
parser.add_argument("--encoder_ff_dropout",metavar='',type=float,default='0.3',help="encoder_ff_dropout ")
parser.add_argument("--pe_max_len",metavar='',type=int,default='5000',help="pe_max_len")

parser.add_argument("--dec_embd_vec_size",metavar='',type=int,default='256',help="dec_embd_vec_size")
parser.add_argument("--decoder_dmodel",metavar='',type=int,default='256',help="decoder_dmodel")
parser.add_argument("--decoder_heads",metavar='',type=int,default='4',help="decoder_heads")
parser.add_argument("--decoder_dinner",metavar='',type=int,default='1024',help="decoder_dinner")
parser.add_argument("--decoder_dropout",metavar='',type=float,default='0.1',help="decoder_dropout ")
parser.add_argument("--decoder_ff_dropout",metavar='',type=float,default='0.3',help="decoder_ff_dropout ")
parser.add_argument("--decoder_layers",metavar='',type=int,default='4',help="decoder_layers ")
parser.add_argument("--tie_dec_emb_weights",metavar='',type=int,default=0,help="tie_dec_emb_weights")
parser.add_argument("--warmup_steps",metavar='',type=int,default=25000,help="warmup_steps")
parser.add_argument("--nepochs",metavar='',type=int,default='100',help="No of epochs")

#---------------------------------------------------------------------------------------------------------
## MT_parameters
parser.add_argument("--encoder_dropout_MT",metavar='',type=float,default='0.3',help="encoder dropout ")
parser.add_argument("--encoder_layers_MT",metavar='',type=int,default='4',help="encoder dropout ")
parser.add_argument("--encoder_dmodel_MT",metavar='',type=int,default='512',help="encoder_dmodel")
parser.add_argument("--encoder_heads_MT",metavar='',type=int,default='4',help="encoder_heads")
parser.add_argument("--encoder_dinner_MT",metavar='',type=int,default='1024',help="encoder_dinner")
parser.add_argument("--encoder_ff_dropout_MT",metavar='',type=float,default='0.3',help="encoder_ff_dropout ")
parser.add_argument("--pe_max_len_MT",metavar='',type=int,default='5000',help="pe_max_len")

parser.add_argument("--dec_embd_vec_size_MT",metavar='',type=int,default='512',help="dec_embd_vec_size")
parser.add_argument("--decoder_dmodel_MT",metavar='',type=int,default='512',help="decoder_dmodel")
parser.add_argument("--decoder_heads_MT",metavar='',type=int,default='4',help="decoder_heads")
parser.add_argument("--decoder_dinner_MT",metavar='',type=int,default='1024',help="decoder_dinner")
parser.add_argument("--decoder_dropout_MT",metavar='',type=float,default='0.1',help="decoder_dropout ")
parser.add_argument("--decoder_ff_dropout_MT",metavar='',type=float,default='0.3',help="decoder_ff_dropout ")
parser.add_argument("--decoder_layers_MT",metavar='',type=int,default='4',help="decoder_layers ")
parser.add_argument("--tie_dec_emb_weights_MT",metavar='',type=int,default=0,help="tie_dec_emb_weights")
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
parser.add_argument("--learning_rate",metavar='',type=float,default='0.0003',help="Value of learning_rate ")
parser.add_argument("--lr_scale",metavar='',type=float,default='1',help="Value of lr_scale ")
parser.add_argument("--step_num",metavar='',type=int,default=1,help="step_num")
parser.add_argument("--clip_grad_norm",metavar='',type=float,default='0',help="Value of clip_grad_norm ")
parser.add_argument("--accm_grad",metavar='',type=float,default='2',help="Value of gradient accumilation, workaround against using larger batch size ")
parser.add_argument("--new_bob_decay",metavar='',type=int,default='0',help="Value of new_bob_decay ") ###remove

#####Loss function parameters
parser.add_argument("--label_smoothing",metavar='',type=float,default='0.1',help="label_smoothing float value 0.1")

####Training schedule parameters 
parser.add_argument("--no_of_checkpoints",metavar='',type=int,default='2',help="Flag of no_of_checkpoints ")
parser.add_argument("--tr_disp",metavar='',type=int,default='100',help="Value of tr_disp ")
parser.add_argument("--vl_disp",metavar='',type=int,default='10',help="Value of vl_disp ")
parser.add_argument("--noise_inj_ratio",metavar='',type=float,default='0.1',help="Value of noise_inj_ratio ")
parser.add_argument("--weight_noise_flag",metavar='',type=str,default=0,help="T|F Flag for weight noise injection")

parser.add_argument("--early_stopping",metavar='',type=int,default=0,help="Value of early_stopping ")
parser.add_argument("--early_stopping_checkpoints",metavar='',type=int,default=5,help="Value of early_stopping_checkpoints ")
parser.add_argument("--early_stopping_patience",metavar='',type=int,default=5,help="Value of early_stopping_patience ")

parser.add_argument("--reduce_learning_rate_flag",metavar='',type=int,default=0,help="reduce_learning_rate_flag True|False") ###remove
parser.add_argument("--lr_redut_st_th",metavar='',type=int,default=3,help="Value of lr_redut_st_th after this epochs the ls reduction gets applied") ###remove
#---------------------------

####bactching parameers
parser.add_argument("--model_dir",metavar='',type=str,default='models/Default_folder',help="model_dir")
parser.add_argument("--batch_size",metavar='',type=int,default='10',help="batch_size")
parser.add_argument("--max_batch_label_len",metavar='',type=int,default='5000',help="max_batch_label_len")
parser.add_argument("--MT_max_batch_label_len",metavar='',type=int,default='5000',help="max_batch_label_len")


parser.add_argument("--max_batch_len",metavar='',type=int,default='2000',help="max_batch_len")
parser.add_argument("--val_batch_size",metavar='',type=int,default='2000',help="val_batch_size")

parser.add_argument("--validate_interval",metavar='',type=int,default='200',help="steps")
parser.add_argument("--max_train_examples",metavar='',type=int,default='200',help="steps")
parser.add_argument("--max_val_examples",metavar='',type=int,default='150',help="steps")

parser.add_argument("--max_feat_len",metavar='',type=int,default='2000',help="max_seq_len the dataloader does not read the sequences longer that the max_feat_len, for memory and some times to remove very long sent for LSTM")
parser.add_argument("--max_label_len",metavar='',type=int,default='200',help="max_labes_len the dataloader does not read the sequences longer that the max_label_len, for memory and some times to remove very long sent for LSTM")

###plot the figures
parser.add_argument("--plot_fig_validation",metavar='',type=int,default=0,help="True|False Not yet implementd")
parser.add_argument("--plot_fig_training",metavar='',type=int,default=0,help="True|False Not yet implementd")

#**********************************
#**********************************
#Spec Aug
parser.add_argument("--spec_aug_flag",metavar='',type=int,default=0,help="spec_aug_flag")
parser.add_argument("--min_F_bands",metavar='',type=int,default='30',help="min_F_bands")
parser.add_argument("--max_F_bands",metavar='',type=int,default='80',help="max_F_bands")
parser.add_argument("--time_drop_max",metavar='',type=int,default='4',help="time_drop_max")
parser.add_argument("--time_window_max",metavar='',type=int,default='4',help="time_window_max")
#**********************************
#---------------------------

#---------------------------
####paths and tokenizers
parser.add_argument("--data_dir",metavar='',type=str,default='/mnt/matylda3/vydana/HOW2_EXP/MT_Transformer/models/Default_folder',help="")
parser.add_argument("--MT_data_dir",metavar='',type=str,default='/mnt/matylda3/vydana/HOW2_EXP/MT_Transformer/models/Default_folder',help="")

parser.add_argument("--src_text_file",metavar='',type=str,default='/mnt/matylda3/vydana/HOW2_EXP/MT_Transformer/Tokenizers/full_text_id.en',help="")
parser.add_argument("--tgt_text_file",metavar='',type=str,default='/mnt/matylda3/vydana/HOW2_EXP/MT_Transformer/Tokenizers/full_text_id.pt',help="")



parser.add_argument("--train_path",metavar='',type=str,default='/mnt/matylda3/vydana/benchmarking_datasets/Timit/scp_files/train/',help="model_dir")
parser.add_argument("--dev_path",metavar='',type=str,default='/mnt/matylda3/vydana/benchmarking_datasets/Timit/scp_files/dev/',help="model_dir")
parser.add_argument("--test_path",metavar='',type=str,default='/mnt/matylda3/vydana/benchmarking_datasets/Timit/scp_files/test/',help="model_dir")
#---------------------------
###ASR tokenizers
parser.add_argument("--Src_model_path",metavar='',type=str,default='/mnt/matylda3/vydana/benchmarking_datasets/Timit/models/Timit_PHSEQ_100/Timit_PHSEQ__100__word.model',help="model_dir")
parser.add_argument("--Tgt_model_path",metavar='',type=str,default='/mnt/matylda3/vydana/benchmarking_datasets/Timit/models/Timit_PHSEQ_100/Timit_PHSEQ__100__word.model',help="model_dir")

####pretrained weights
#---------------------------
parser.add_argument("--pre_trained_weight",metavar='',type=str,default='0',help="pre_trained_weight if you dont have just give zero ")
parser.add_argument("--retrain_the_last_layer",metavar='',type=str,default='False',help="retrain_final_layer if you dont have just give zero ")
####load the weights
#---------------------------------------------------------------------------------------------------------------------
parser.add_argument("--pre_trained_MT_weight",metavar='',type=str,default='0',help="pre_trained_weight if you dont have just give zero ")
parser.add_argument("--pre_trained_ASR_weight",metavar='',type=str,default='0',help="pre_trained_weight if you dont have just give zero ")

parser.add_argument("--MT_interpolation", metavar='', type=float, default=0.5, help="MT_interpolation;  if you dont have just give 0.5 ")
parser.add_argument("--ASR_interpolation",metavar='', type=float, default=0.5, help="ASR_interpolation; if you dont have just give 0.5 ")

parser.add_argument("--ASR_model",metavar='',type=str,default='0',help="will be a updated in code to list of models for ensembling")
parser.add_argument("--MT_model",metavar='',type=str,default='0',help="will be a updated in code to list of models for ensembling")



parser.add_argument("--ASR_LM_model",metavar='',type=int,default=0,help="will be a updated in code to list of models for ensembling")
parser.add_argument("--MT_LM_model",metavar='',type=int,default=0,help="will be a updated in code to list of models for ensembling")

parser.add_argument("--ASR_LM_model_path",metavar='',type=str,default='0',help="will be a updated in code to list of models for ensembling")
parser.add_argument("--MT_LM_model_path",metavar='',type=str,default='0',help="will be a updated in code to list of models for ensembling")

parser.add_argument("--ASR_LM_model_weight",metavar='',type=float,default=1,help="will be a updated in code to list of models for ensembling")
parser.add_argument("--MT_LM_model_weight",metavar='',type=float,default=1,help="will be a updated in code to list of models for ensembling")


#--------------------------------------------------------------------------------------------------------------------
#---------------------------
parser.add_argument("--weight_text_file",metavar='',type=str,default='weight_folder/weight_file',help="weight_file")
parser.add_argument("--Res_text_file",metavar='',type=str,default='weight_folder/weight_file_res',help="Res_file")
#----------------------------
parser.add_argument("--MT_flag",metavar='',type=int,default=0,help="MT_flag")


####decoding_parameters
parser.add_argument("--RNNLM_model",metavar='',type=str,default='/mnt/matylda3/vydana/HOW2_EXP/Timit/models/TIMIT_fullnewsetup_2_4dr0.3_LAS_loc_arg_format_V2/model_architecture_',help="")
parser.add_argument("--LM_model",metavar='',type=str,default='None',help="LM_model")
parser.add_argument("--Am_weight",metavar='',type=float,default=1,help="lm_weight a float calue between 0 to 1 --->(Am_weight* Am_pred + (1-Am_weight)*lm_pred)")
parser.add_argument("--ASR_beam",metavar='',type=int,default=10,help="beam for decoding")
parser.add_argument("--MT_input_beam",metavar='',type=int,default=10,help="beam for decoding")
parser.add_argument("--MT_beam",metavar='',type=int,default=10,help="beam for decoding")
parser.add_argument("--gamma",metavar='',type=float,default=1,help="gamma (0-2), noisy eos rejection scaling factor while decoding")
parser.add_argument("--ASR_len_pen",metavar='',type=float,default=1,help="len_pen(0.5-2), len_pen maximum number of decoding steps")
parser.add_argument("--MT_len_pen",metavar='',type=float,default=1,help="len_pen(0.5-2), len_pen maximum number of decoding steps")
parser.add_argument("--ASR_len_bonus",metavar='',type=float,default=1,help="len_pen(0.5-2), len_pen maximum number of decoding steps")
parser.add_argument("--MT_len_bonus",metavar='',type=float,default=1,help="len_pen(0.5-2), len_pen maximum number of decoding steps")
parser.add_argument("--use_Just_ASR_MT_tokens",metavar='',type=float,default=0,help="len_pen(0.5-2), len_pen maximum number of decoding steps")

parser.add_argument("--finetune_with_Puncmodel",metavar='',type=float,default=0,help="flag to find to punctuvate with the model finetune_with_Puncmodel")

parser.add_argument("--encoder_layers_Punc",metavar='',type=int,default=1,help="flag to find to punctuvate with the model layers")
parser.add_argument("--decoder_layers_Punc",metavar='',type=int,default=1,help="flag to find to punctuvate with the model layers")






parser.add_argument("--Decoding_job_no",metavar='',type=int,default=0,help="Res_file")
parser.add_argument("--scp_for_decoding",metavar='',type=int,default=0,help="scp file for decoding")
parser.add_argument("--plot_decoding_pics",metavar='',type=int,default=0,help="T|F")
parser.add_argument("--decoder_plot_name",metavar='',type=str,default='default_folder',help="T|F")

parser.add_argument("--SWA_random_tag",metavar='',type=int,default='0',help="T|F")
#---------------------------
parser.add_argument("-v","--verbosity",action="count",help="increase output verbosity")






