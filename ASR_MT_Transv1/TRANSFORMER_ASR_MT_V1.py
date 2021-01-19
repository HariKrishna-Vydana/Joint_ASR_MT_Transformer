import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb 
from torch.autograd import Variable
import kaldi_io


import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils.weight_norm as wtnrm

import numpy as np

import sys
sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/Joint_ASR_MT_Transformer/ASR_MT_Transv1')

from CMVN import CMVN
from Trans_conv_layers import Conv_2D_Layers
from Trans_Decoder import Decoder
from Trans_Encoder import Encoder




#--------------------------------------------------------------------------
class Transformer(nn.Module):
    """An encoder-decoder framework only includes attention. """
    def __init__(self,args):
        super(Transformer, self).__init__()   
        "Defines the Joint ASR-MT training model, MT_flag is False if the model uses speech input and MT_flag True if the model"
        #breakpoint()
        self.conv_layers = Conv_2D_Layers(args)
        self.ASR_encoder = Encoder(args=args,MT_flag=False)
        self.ASR_decoder = Decoder(args=args,MT_flag=False)

        self.MT_encoder = Encoder(args=args,MT_flag=True)
        self.MT_decoder = Decoder(args=args,MT_flag=True)
        #----------------------------------
    def forward(self,padded_Src_speech,padded_Src_seq,padded_Tgt_seq):
        ###conv layers
        #General Transformer MT model
        conv_padded_Src_seq = self.conv_layers(padded_Src_speech)

        encoder_padded_outputs, *_ = self.ASR_encoder(conv_padded_Src_seq)
        ASR_output_dict = self.ASR_decoder(padded_Src_seq, encoder_padded_outputs)

        #print(ASR_output_dict.get('dec_output').shape)
        
        MT_encoder_padded_outputs, *_ = self.MT_encoder(ASR_output_dict.get('dec_output'))
        MT_output_dict = self.MT_decoder(padded_Tgt_seq,MT_encoder_padded_outputs)
        
        #print(ASR_output_dict.keys(), MT_output_dict.keys())
        #print(MT_output_dict.get('dec_output').shape)

        MT_output_dict['cost'] = MT_output_dict.get('cost') + ASR_output_dict.get('cost')
        return MT_output_dict
    #=============================================================================================================
    #=============================================================================================================
    #==============================================================================
    def predict(self,feat_path,args):
        print("went to the decoder loop")
        with torch.no_grad():
                
                print(feat_path) 
                #### read feature matrices 
                smp_feat=kaldi_io.read_mat(feat_path)
                print(smp_feat.shape)
                if args.apply_cmvn:
                       smp_feat=CMVN(smp_feat)
 
                input=torch.from_numpy(smp_feat)
                input = Variable(input.float(), requires_grad=False).double().float()
                input=input.unsqueeze(0) 

                                
        
                #Jointly-Trained Transformer ASR model 
                Decoder_output_dict_init = {'ys':None, 'score_1':None, 'dec_output':None}

                conv_padded_Src_seq = self.conv_layers(input)
                encoder_padded_outputs, *_ = self.ASR_encoder(conv_padded_Src_seq)
                ASR_output_dict = self.ASR_decoder.recognize_batch_beam_autoreg_LM_multi_hyp(Decoder_output_dict_init, encoder_padded_outputs,args.beam,args.Am_weight,args.gamma,args.LM_model,args.len_pen,args) 
                 
                MT_encoder_padded_outputs, *_ = self.MT_encoder(ASR_output_dict.get('dec_output'))
                MT_output_dict = self.MT_decoder.recognize_batch_beam_autoreg_LM_multi_hyp(ASR_output_dict,MT_encoder_padded_outputs,args.beam,args.Am_weight,args.gamma,args.LM_model,args.len_pen,args)
                 
                #Picking the best
                nbest_hyps = MT_output_dict.get('ys')
                scoring_list = MT_output_dict.get('score_1')

                A1,A2=torch.topk(torch.sum(scoring_list, dim=1, keepdim=True), args.beam, dim=0, largest=True, sorted=True)
                
                nbest_hyps = nbest_hyps[A2.squeeze(0)]
                scoring_list = scoring_list[A2.squeeze(0)]

                #print(MT_output_dict[A2.squeeze(0)])
                #print(MT_output_dict.keys())
                #===================================================================================
                beam_len = nbest_hyps.size(0)
                hyp = {'score': 0.0, 'yseq': None,'state': None, 'alpha_i_list':None, 'Text_seq':None}

                #===============================================
                Output_dict=[]
                for I in range(beam_len):    

                    new_hyp={}
                    new_hyp['yseq'] = nbest_hyps[I]
                    new_hyp['score'] = scoring_list[I].sum()
                   
                    #new_hyp['Text_seq'] = self.MT_decoder.get_charecters_for_sequences(nbest_hyps[I])
                    new_hyp['Text_seq'] = self.MT_decoder.get_charecters_for_sequences(nbest_hyps[I],self.MT_decoder.Tgt_model,self.MT_decoder.pad_index,self.MT_decoder.eos_id,self.MT_decoder.word_unk)

                    new_hyp['state'] = hyp['state']
                    new_hyp['alpha_i_list'] = hyp['alpha_i_list']

                    Output_dict.append(new_hyp)
        return Output_dict
        #----------------------------------------------------------------



#=============================================================================================================
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------

#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
class TransformerOptimizer(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, k, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.k = k
        
        #present_lr=[param_group['lr'] for param_group in self.optimizer.param_groups]
        self.init_lr = d_model ** (-0.5)
        self.warmup_steps = warmup_steps
        self.step_num = 0
        self.reduction_factor=1
    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._update_lr()
        self.optimizer.step()

    def _update_lr(self):
        self.step_num += 1
        lr = self.k * self.init_lr * min(self.step_num ** (-0.5),
                                         self.step_num * (self.warmup_steps ** (-1.5)))
        
        #print(lr,self.step_num ** (-0.5),self.step_num * self.warmup_steps ** (-1.5),self.reduction_factor)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()

    def set_k(self, k):
        self.k = k

    def set_step_num(self, step_num):
        self.step_num=step_num

    def reduce_learning_rate(self, k):
        self.reduction_factor = self.reduction_factor*k
        #print(self.reduction_factor)
    
    def print_lr(self):
        present_lr=[param_group['lr'] for param_group in self.optimizer.param_groups]
        return present_lr[0]


#=============================================================================================================

#---------------------------------------------------------------------------------------------------------------
#===============================================================================================================
#===============================================================================================================
#===============================================================================================================
#===============================================================================================================
#===============================================================================================================

