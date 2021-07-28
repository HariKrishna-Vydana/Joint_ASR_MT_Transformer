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
sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/Joint_ASR_MT_Transformer')
from ASR_MT_Transv1.CMVN import CMVN
from ASR_MT_Transv1.Trans_conv_layers import Conv_2D_Layers
from ASR_MT_Transv1.Trans_Decoder import Decoder
from ASR_MT_Transv1.Trans_Encoder import Encoder
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
        #breakpoint()
        
        MT_utterances = torch.sum(torch.sum(padded_Src_speech,dim=1,keepdim=True),dim=2,keepdim=True)==0

        Train_only_ASR = torch.sum(padded_Tgt_seq,dim=1)==0
        All_utt_are_ASR = torch.sum(torch.sum(padded_Tgt_seq,dim=1),dim=0)==0

        ####remove loss if no MT trans is included
        padded_Tgt_seq[Train_only_ASR] == self.MT_decoder.IGNORE_ID


        conv_padded_Src_seq = self.conv_layers(padded_Src_speech)
        encoder_padded_outputs, *_ = self.ASR_encoder(conv_padded_Src_seq)
        encoder_padded_outputs = encoder_padded_outputs * MT_utterances * 1
        ASR_output_dict = self.ASR_decoder(padded_Src_seq, encoder_padded_outputs)

        #=================================================================================
        if All_utt_are_ASR:
            MT_output_dict = ASR_output_dict

        else:

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
                Decoder_output_dict_init = {'ys':None, 'score_1':None, 'dec_output':None,'input':input}
                
                # breakpoint()
                conv_padded_Src_seq = self.conv_layers(input)
                encoder_padded_outputs, *_ = self.ASR_encoder(conv_padded_Src_seq)
                ASR_output_dict = self.ASR_decoder.recognize_batch_beam_autoreg_LM_multi_hyp(Decoder_output_dict_init, encoder_padded_outputs, args.ASR_beam, args.Am_weight, args.gamma, args.LM_model, args.ASR_len_pen, args) 


                ASR_nbest = ASR_output_dict.get('ys')
                ASR_output_text_seq={}
                for I in range(ASR_nbest.size(0)):
                    ASR_output=self.ASR_decoder.get_charecters_for_sequences(ASR_nbest[I],self.ASR_decoder.Tgt_model,self.ASR_decoder.pad_index,self.ASR_decoder.eos_id,self.ASR_decoder.word_unk)
                    ASR_output_text_seq[I]=ASR_output
                ####
                ASR_output_dict['ASR_output_text_seq']=ASR_output_text_seq
                #-------------------------------------------------------------
                #-------------------------------------------------------------
                #-------------------------------------------------------------


                MT_encoder_padded_outputs, *_ = self.MT_encoder(ASR_output_dict.get('dec_output'))
                MT_output_dict = self.MT_decoder.recognize_batch_beam_autoreg_LM_multi_hyp(ASR_output_dict, MT_encoder_padded_outputs, args.MT_beam, args.Am_weight, args.gamma, args.LM_model, args.MT_len_pen, args)
                

                #Picking the best
                nbest_hyps = MT_output_dict.get('ys')
                scoring_list = MT_output_dict.get('score_1')
                A1,A2=torch.topk(torch.sum(scoring_list, dim=1, keepdim=True), args.MT_beam, dim=0, largest=True, sorted=True)
                
                nbest_hyps = nbest_hyps[A2.squeeze(0)]
                scoring_list = scoring_list[A2.squeeze(0)]

                #breakpoint()
                #print(MT_output_dict[A2.squeeze(0)])
                #print(MT_output_dict.keys())
                #===================================================================================
                beam_len = nbest_hyps.size(0)
                hyp = {'score': 0.0, 'yseq': None,'state': None, 'alpha_i_list':None, 'Text_seq':None}

                #===============================================
                ASR_MT_Output_dict={}
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

                ASR_MT_Output_dict['MT_Output_dict']=Output_dict
                ASR_MT_Output_dict['ASR_output_dict']=ASR_output_dict
        return ASR_MT_Output_dict
        #----------------------------------------------------------------



#=============================================================================================================
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------

#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
class TransformerOptimizer(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, k, d_model,step_num,warmup_steps=4000):
        self.optimizer = optimizer
        self.k = k
        

        self.init_lr = d_model ** (-0.5)
        self.warmup_steps = warmup_steps

        self.step_num = step_num
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

