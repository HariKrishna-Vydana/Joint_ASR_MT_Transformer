import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb 
from torch.autograd import Variable

import sys
sys.path.insert(0, '/mnt/matylda3/vydana/HOW2_EXP/Joint_ASR_MT_Transformer')
from ASR_MT_Transv1.Trans_utilities import get_attn_key_pad_mask, get_subsequent_mask, get_attn_pad_mask_encoder, get_attn_pad_mask,get_encoder_non_pad_mask, get_decoder_non_pad_mask,pad_list
from ASR_MT_Transv1.Trans_MHA import MultiHeadAttention, PositionwiseFeedForward, PositionalEncoding
from ASR_MT_Transv1.Load_sp_model import Load_sp_models
#===========================================================================================================

class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1,ff_dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=ff_dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):

        x=dec_input

        nx=self.norm1(x)
        dec_output, dec_slf_attn = self.slf_attn(nx, nx,nx, mask=slf_attn_mask)
        x=x+self.dropout(dec_output)

        nx=self.norm2(x)
        dec_output, dec_enc_attn = self.enc_attn(nx, enc_output, enc_output, mask=dec_enc_attn_mask)
        x=x+self.dropout(dec_output)

        nx=self.norm3(x)
        dec_output = self.pos_ffn(nx)
        dec_output=x+self.dropout(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn

#====================================================
#----------------------------------------------------
class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self,args,MT_flag, Punc_flag):
        super(Decoder, self).__init__()
        
        #------------------------------------------------------------
        self.MT_flag = MT_flag
        self.Punc_flag = Punc_flag        

        self.Tgt_model_path = args.Tgt_model_path if self.MT_flag else args.Src_model_path
        print(self.Tgt_model_path)
        ####word model
        self.Tgt_model = Load_sp_models(self.Tgt_model_path)

        self.targets_no = int(self.Tgt_model.__len__())
        self.pad_index  = self.targets_no       
        self.sos_id     = self.targets_no + 1 # Start of Sentence
        self.eos_id     = self.targets_no + 2 # End of Sentence
        self.mask_id    = self.targets_no + 3
        self.Wout_size  = self.targets_no + 4
        self.word_unk   = self.Tgt_model.unk_id()
        self.Word_SIL_tok   = self.Tgt_model.EncodeAsIds('_____')[0]
        self.IGNORE_ID = self.pad_index
        self.n_tgt_vocab = self.Wout_size
        #---------------------------------------------------------------
        # parameters
        self.d_word_vec = args.dec_embd_vec_size_MT if self.MT_flag else args.dec_embd_vec_size

        self.n_layers = args.decoder_layers_MT if self.MT_flag else args.decoder_layers
        self.n_layers = args.decoder_layers_Punc if self.Punc_flag else self.n_layers        


        self.n_head = args.decoder_heads_MT    if self.MT_flag else args.decoder_heads
        self.d_model = args.decoder_dmodel_MT if self.MT_flag else args.decoder_dmodel
        self.d_inner = args.decoder_dinner_MT if self.MT_flag else args.decoder_dinner
        self.dropout = args.decoder_dropout_MT if self.MT_flag else args.decoder_dropout
        self.ff_dropout = args.decoder_ff_dropout_MT if self.MT_flag else args.decoder_ff_dropout
        self.tie_dec_emb_weights = args.tie_dec_emb_weights_MT if self.MT_flag else args.tie_dec_emb_weights
        self.pe_max_len = args.pe_max_len_MT if self.MT_flag else args.pe_max_len
        #---------------------------------------------------------------

        self.d_k = int(self.d_model/self.n_head)
        self.d_v = int(self.d_model/self.n_head)

        self.tgt_word_emb = nn.Embedding(self.n_tgt_vocab, self.d_word_vec)
        self.x_scale = math.sqrt(self.d_model)       

        self.positional_encoding = PositionalEncoding(self.d_model, max_len=self.pe_max_len, dropout=self.dropout)
        self.output_norm=nn.LayerNorm(self.d_model)

        self.layer_stack = nn.ModuleList([DecoderLayer(self.d_model, self.d_inner, self.n_head,
                                                       self.d_k, self.d_v, dropout=self.dropout,
                                                       ff_dropout=self.ff_dropout) for _ in range(self.n_layers)])
        self.tgt_word_prj = nn.Linear(self.d_model, self.n_tgt_vocab)

        ###weight tie-ing 
        ##for the tieing to be possible weights of self.tgt_word_emb, self.tgt_word_prj should be equal 
        ### self.d_word_vec should be equal to self.d_model or a linear layer should be used
        if self.tie_dec_emb_weights:
                self.tgt_word_emb.weight=self.tgt_word_prj.weight

        ###label_smoothed_cross_entropy
        self.label_smoothing=args.label_smoothing
     

    def preprocess(self, padded_input):
        """Generate decoder input and output label from padded_input
        Add <sos> to decoder input, and add <eos> to decoder output label """

        ys = [y[y != self.IGNORE_ID] for y in padded_input]  # parse padded ys
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos_id])
        sos = ys[0].new([self.sos_id])
        ys_in = [torch.cat([sos, y], dim=0) for y in ys ]
       

        ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos_id)
        ys_out_pad = pad_list(ys_out, self.IGNORE_ID) ####original
        
        assert ys_in_pad.size() == ys_out_pad.size()
        return ys_in_pad, ys_out_pad

    def forward(self, padded_input, encoder_padded_outputs,return_attns=True):
        """ Args: padded_input: N x To encoder_padded_outputs: N x Ti x H   Returns:"""
        ##################################################################### 
        # breakpoint()    
        cuda_flag = encoder_padded_outputs.is_cuda
        device_flag = torch.device("cuda") if cuda_flag else torch.device("cpu")
        #===================================================================   
        dec_slf_attn_list, dec_enc_attn_list = [], []
        #Get Deocder Input and Output
        ys_in_pad, ys_out_pad = self.preprocess(padded_input)
        
        #Prepare masks
        slf_attn_mask_subseq = get_subsequent_mask(ys_in_pad)
        output_length = ys_in_pad.size(1)

        ###Not using other masks, using Justs casual masks
        non_pad_mask = None;
        dec_enc_attn_mask = None;
        slf_attn_mask_keypad = None
        slf_attn_mask = slf_attn_mask_subseq

        embd_output = self.tgt_word_emb(ys_in_pad)
        dec_output = self.positional_encoding(embd_output) ####has dropout inside by 0.1

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(dec_output, encoder_padded_outputs,
                                                                non_pad_mask=non_pad_mask,slf_attn_mask=slf_attn_mask,
                                                                dec_enc_attn_mask=dec_enc_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        # before softmax
        #------------
        dec_output=self.output_norm(dec_output)
        seq_logit = self.tgt_word_prj(dec_output)
        #---------------
        pred, gold = seq_logit, ys_out_pad
        
        cost, CER = cal_performance(pred, gold,self.IGNORE_ID,normalize_length=False,smoothing=self.label_smoothing)
        
        #breakpoint()
        # output_dict={'cost':cost, 'CER':CER, 'smp_pred':pred,'smp_gold':gold}       
        output_dict = {'cost':cost, 'dec_slf_attn_list':dec_slf_attn_list, 'dec_enc_attn_list':dec_enc_attn_list,'dec_output':dec_output, 'Char_cer':CER, 'Word_cer':CER}
        return output_dict


#==========================================================================
#==========================================================================
    def prediction_from_trained_model(self,ys,encoder_outputs,scores_list,):
                """####this function is accessed from the decoder to get the output from the decoder,
                   and this could be used for model ensembling an
                   ####when this function is called with prediceted label sequences,
                   it gives the proability distribution for the next possible labels roughly this gives P(y_i |y_(i<i)
                """
                non_pad_mask = torch.ones_like(ys).float().unsqueeze(-1) # 1xix1
                slf_attn_mask = get_subsequent_mask(ys)

                # -- Forward
                #dec_output=self.positional_encoding(ys)
                dec_output=self.positional_encoding(self.tgt_word_emb(ys))

                for dec_layer in self.layer_stack:
                    dec_output, _, _ = dec_layer(dec_output, encoder_outputs,non_pad_mask=None,slf_attn_mask=slf_attn_mask,dec_enc_attn_mask=None)


                dec_output_Bneck=self.output_norm(dec_output)

                dec_output=self.output_norm(dec_output[:, -1])
                seq_logit = self.tgt_word_prj(dec_output)

                scores_list.append(seq_logit.unsqueeze(1))
                local_scores = F.log_softmax(seq_logit, dim=1)
                scores = F.softmax(seq_logit, dim=1)
                present_label=torch.argmax(scores, dim=1)
                return local_scores,scores_list,present_label,dec_output_Bneck
#=============================================================================================================
##============================================================================================================
#-------------------------------------------------------------------------------------------------------------
    def get_multiple_hypothesis(self,store_ended_hyps,store_ended_LLR,ys,score_1,i,maxlen):

        """ *******works for batch one only*******
        While dong the beam serch there is a scenario where one of the prefix has ended and others have not,
        Now this prefix kills other hypotheis,this happens as it gets zeros score for predicting next-label,
        after eos and any prediction after eos is mapped to eos.
        #
        To avoid this and increase divesity in the prefixes, in the beam the prefixes which have already had the eos will be given a very large -ve score
        this function does that task;
        """

        present_list=[];present_llr=[]
        #works for batch_one_only
        start_collecting=False
        #-------------------------
        #the first token should not be EOS
        mov_id=ys[:,-1]==self.eos_id
        #-------------------------
        if (i==maxlen-1): 
            collecting_ys=ys    
            collecting_score=score_1
            start_collecting=True

        #-------------------------------
        elif len(ys[mov_id])>0:
            collecting_ys=ys[mov_id]
            collecting_score=score_1[mov_id]
            start_collecting=True
            
            ###for pruning out theprefixes which will go to list
            score_1[mov_id]=score_1[mov_id]-1000
        else:
            pass

        #-------------------------------
        if start_collecting:
            [present_list.append(i.squeeze()) for i in torch.split(collecting_ys,1,dim=0)]
            [present_llr.append(i.squeeze()) for i in torch.split(collecting_score,1,dim=0)]
            store_ended_hyps += present_list
            store_ended_LLR += present_llr
            start_collecting=False

        #-------------------------------
        return score_1,store_ended_hyps,store_ended_LLR
#=============================================================================================================
##############################################################################################################
    def Regrouping_bottleneck_features(self,dec_output_Bneck_org,beam,Selc_Index,batch_size,hyps):
                """ For the Joint model training the bottle-necks from the model are to be collected for every prefix 
                    as the prefixes get pruned out in beam search the associated bottlencks need to be pruned out and,
                    the ones that finally stand-out in the beam search has the corresponding bottlenecks in this outpout
                """
                dec_output_Bneck=dec_output_Bneck_org
                dec_output_Bneck=dec_output_Bneck.view(batch_size,hyps,dec_output_Bneck.size(1),dec_output_Bneck.size(2))
                rep_dec_output_Bneck=torch.repeat_interleave(dec_output_Bneck,beam,1)
                Bneck_select=torch.cat([Selc_Index.unsqueeze(3)]*rep_dec_output_Bneck.size(3),dim=3)
                Bneck_select=torch.cat([Bneck_select]*rep_dec_output_Bneck.size(2),dim=2)
                #Bneck_select=torch.cat([Bneck_select]*rep_dec_output_Bneck.size(0),dim=0)
                #Bneck_select=torch.cat([Bneck_select]*beam,dim=1)
                ASR_dec_output=torch.gather(rep_dec_output_Bneck,1,Bneck_select)
                ASR_dec_output=ASR_dec_output.view(batch_size*hyps,-1,dec_output_Bneck_org.size(2))
                return ASR_dec_output
################################################################################################################
    def prediction_from_trained_model_beam_Search(self,i,ys,score_1,AM_local_scores,beam,hyps,gamma,batch_size):
            """
            ####vecotorized beam-search ===>beam search that happens parllelly i.e., 
            1.Each prefix is treated as a invidual sequence when given to the model and the predictions for each prefixes are obtained;
            2.Each prefix has a beam of new possible labels, so each prefix is repeated beam number of times and the new label is concatented so does the likeli-hood score;
            3.the new prefixes are hyps_no*beam are pruned to settle with hyps_no prefixes
            4.Eos threshold is used if any of the predicted labels in the beam is eos
            5. If any of the hypotheis has ended the duplication avoided to increase diverse batches

            #folded accordingly and the beam of new
            """

            if i==0:

                ###for the first time just repeat the hyps and add the beam to the hyposhesis 
                local_best_scores, local_best_ids = torch.topk(AM_local_scores, hyps, dim=1,largest=True,sorted=True)
                #---------------------
                present_ids=(local_best_ids[::hyps]).contiguous().view(-1,1)
                present_scores=(local_best_scores[::hyps]).contiguous().view(-1,1)
            
                ##for not allow ing eos as first token 
                ##first lable cannot be eos
                #-----------------------------------------------------------------------
                mask=torch.eq(present_ids,self.eos_id)

                ys=torch.cat((ys,present_ids),dim=1)
                score_1=torch.cat((score_1,present_scores),dim=1)
                #-----------------------------------------------------------------------                            
                mask=torch.eq(present_ids,self.eos_id)
                score_1=score_1-mask*1000


                ###Not corrected ------>should be expanded and selected with selection index,,,,, but model regenerates them with labels in i=>1
                #ASR_dec_output=dec_output_Bneck_org
            #----------------------------------------------------------------------------    
            else:

                #---------------------
                local_best_scores, local_best_ids = torch.topk(AM_local_scores, beam, dim=1,largest=True,sorted=True)
                #---------------------               
                ###################################################
                

                ####filtering EOS if EOS has occured with the value leess than the threshold then filtering out 
                # ---------EoS threshold--------------------------------
                not_eos_mask=(local_best_ids==self.eos_id)
                ###EOS scores and ids ,Non EOS score and IDs
                ##max of Non Eos in dim=1 by making non-Eos -1000
                ##max of Eos in dim=1 by making non Eos -1000
                ##compute the threshold if [ EOS > gamma * NON_EOS]
                #####filter out using outer product

                NON_EOS_mask,NON_EOS_mask_ids=torch.max(local_best_scores*~not_eos_mask +not_eos_mask*-1000,dim=1)
                EOS_mask,EOS_mask_ids=torch.max(local_best_scores*not_eos_mask +~not_eos_mask*-1000,dim=1)               

                EOS_out=EOS_mask>gamma*NON_EOS_mask

                EOS_SCORE_MASK=(not_eos_mask.transpose(0,1)*EOS_out).transpose(0,1)
                local_best_scores=local_best_scores-(not_eos_mask*1*~EOS_SCORE_MASK*1000.0)        
                #--------------------------------------------------------              
                #repeat the prefixes beam times

                ys_1=torch.repeat_interleave(ys,beam,0)
                score_2=torch.repeat_interleave(score_1,beam,0)
                #----------------------------------------------------
                present_ids=(local_best_ids).contiguous().view(-1,1)
                present_scores=(local_best_scores).contiguous().view(-1,1)
                #----------------------------------------------------
                #concatenate labels and scores to the prefixes
                ys=torch.cat((ys_1,present_ids),dim=1)
                score_1=torch.cat((score_2,present_scores),dim=1)
                #----------------------------------------------------

                ###fold accordingly to get hyps *beam and prune out the worst hypothisis keeping beam no of prefixes
                pres_acuml_score=torch.cumsum(score_1,dim=1)[:,-1]
                al1,al2=torch.topk(pres_acuml_score.view(batch_size,hyps*beam,1),hyps,dim=1,largest=True,sorted=True)
                selecting_index=torch.cat([al2]*ys.size(1),dim=2)
                #----------------------------------------------------
                #----------------------------------------------------
                #filting the bottlenecs as per topk
                #Selc_Index=al2;
                #ASR_dec_output=self.Regrouping_bottleneck_features(dec_output_Bneck_org,beam,Selc_Index,batch_size,hyps)
                #----------------------------------------------------
                ###regrouping acording to utterances after selecting top K this is needed for gathering as per topk
                ys=ys.view(batch_size,hyps*beam,-1)
                score_1=score_1.view(batch_size,hyps*beam,-1)               
                #----------------------------------------------------
                ###prunng the output using gather
                #selecting the top labels and scores
                ys=torch.gather(ys,1,selecting_index)
                score_1=torch.gather(score_1,1,selecting_index)

                ###making it ready for next iteration
                ### converting the selected hypothesis per utterances to the seperate hypothesis to process the parallel
                ys=ys.view(batch_size*hyps,-1)
                score_1=score_1.view(batch_size*hyps,-1)
                ####################################################
                #----------Eos servived the past iteration then it is 
                #acccepted EOS so no new labels after EOS, score, should be set to zero otherwise we get bad hypotheis
                if i>1:
                     selected_EOS=torch.eq(ys[:,-2],self.eos_id)
                     score_1[:,-1]=score_1[:,-1]*(~selected_EOS)
                     ys[:,-1][selected_EOS]=self.eos_id
                #------------------------------
            return ys,score_1
##=============================================================================================================
##======================================================================================================
##======================================================================================================
    def recognize_batch_beam_autoreg_LM_multi_hyp(self, output_dict, encoder_outputs, beam,Am_weight,gamma,LM_model,len_pen,args):
        """Beam search, decode one utterence now. 
        Args: encoder_outputs: T x H, 
        char_list: list of character, args: args.beam, 
        Returns: nbest_hyps: """
        
        enc_out_len = encoder_outputs.size(1)       
        #----------------------------
        maxlen=int(enc_out_len*len_pen)

        len_bonus = args.MT_len_bonus if self.MT_flag else args.ASR_len_bonus

        use_Just_ASR_MT_tokens = args.use_Just_ASR_MT_tokens
        #use_Just_ASR_MT_tokens = 


        ##flags for ASR or MT ensemblings
        Do_MT_Ensembling = ((self.MT_flag==True) and (not isinstance(args.MT_model,str))) and (args.MT_interpolation!=0) 
        Do_ASR_Ensembling = ((self.MT_flag==False) and (not isinstance(args.ASR_model,str))) and (args.ASR_interpolation!=0)

        print(self.MT_flag,isinstance(args.ASR_model,str), args.ASR_interpolation,Do_ASR_Ensembling)
       

        ### This works but can be increased but it takes memory, can be increased Works ---Memory?
        #hyps = args.MT_beam * args.MT_input_beam if self.MT_flag else args.ASR_beam
        hyps=beam
        #----------------------------
        print("beam,hyps,len_pen,maxlen,enc_out_len,Am_weight",beam,hyps,len_pen,maxlen,enc_out_len,Am_weight)        
        batch_size = encoder_outputs.size(0)
       
        ys = torch.ones(batch_size*hyps,1).fill_(self.sos_id).type_as(encoder_outputs).long()                
        if torch.is_tensor(output_dict.get('score_1')):
                ####MT decoding for joint model
                score_1 = torch.sum(output_dict.get('score_1'),dim=1,keepdim=True)
                score_1 = torch.repeat_interleave(score_1,hyps,0)                
        else:
                score_1 = torch.zeros_like(ys).float()


        rep_encoder_outputs=torch.repeat_interleave(encoder_outputs,hyps,0)
        #============================
        if Do_MT_Ensembling:
                Ens_MT_model = args.MT_model
                ##need to chance eos to padding tokens
                MT_input_for_ens = output_dict['ys']

                MT_input_for_ens = MT_input_for_ens[:,1:]
                MT_input_for_ens[MT_input_for_ens == Ens_MT_model.decoder.eos_id] = Ens_MT_model.decoder.IGNORE_ID

                Ens_MT_model_encoder_outputs, *_= Ens_MT_model.encoder(MT_input_for_ens)
                rep_Ens_MT_model_encoder_outputs=torch.repeat_interleave(Ens_MT_model_encoder_outputs,hyps,0)


        if Do_ASR_Ensembling:
                Ens_ASR_model = args.ASR_model
                Ens_ASR_conv_padded_Src_seq = Ens_ASR_model.conv_layers(output_dict['input'])
                Ens_ASR_model_encoder_outputs, *_ = Ens_ASR_model.encoder(Ens_ASR_conv_padded_Src_seq)
                rep_Ens_ASR_model_encoder_outputs = torch.repeat_interleave(Ens_ASR_model_encoder_outputs,hyps,0)

                # print("not implemented")


        #print(args.ASR_LM_model, args.MT_LM_model)
        LM_model = args.MT_LM_model if self.MT_flag else args.ASR_LM_model 
        Am_weight = args.MT_LM_model_weight if self.MT_flag else args.ASR_LM_model_weight

        #============================
        store_ended_hyps = []
        store_ended_LLR = []
        #============================
        scores_list=[]
        start_collecting=False
        for i in range(maxlen): 
            #----------------------------------------------------  
            ## if loop to use or not an LM (or) skip the LM for the first step
            ## 

            if use_Just_ASR_MT_tokens:

                    COMB_AM_MT_local_scores=0 #####just for keeping the comination equaion intact under Do_MT_Ensembling, Do_ASR_Ensembling
            else:

                if Am_weight==1: 
                    #print("not using a LM")
                    COMB_AM_MT_local_scores, scores_list, present_label, dec_output_Bneck = self.prediction_from_trained_model(ys,rep_encoder_outputs,scores_list)        
                    ###should not be used but just dummy bottleneck
                    ####dec_output_Bneck features always from joint acoustic model

                else:
                    AM_local_scores,scores_list,present_label,dec_output_Bneck=self.prediction_from_trained_model(ys,rep_encoder_outputs,scores_list)
                    COMB_AM_MT_local_scores=AM_local_scores
                   
                    ####usuvally lm fusion is done here but as we are doing ensembling we first do ensemble and add LM later
            

            #-------------------------------------------------------------------------------------------------------------------------          
            ###Could be removed tf there is not ensembling
            #Ensembling ASR or MT models

            if Do_MT_Ensembling:    
                    Ens_MT_model_predict_prob, _, _, _ = Ens_MT_model.decoder.prediction_from_trained_model(ys, rep_Ens_MT_model_encoder_outputs, scores_list)                
                    COMB_AM_MT_local_scores = Ens_MT_model_predict_prob*args.MT_interpolation + COMB_AM_MT_local_scores *(1-args.MT_interpolation)

            #-------
            if Do_ASR_Ensembling:
                    Ens_ASR_model_predict_prob, _, _, _ = Ens_ASR_model.decoder.prediction_from_trained_model(ys, rep_Ens_ASR_model_encoder_outputs, scores_list)
                    COMB_AM_MT_local_scores = Ens_ASR_model_predict_prob*args.ASR_interpolation + COMB_AM_MT_local_scores*(1-args.ASR_interpolation)
            #-------

            #---------------------------------------------------------------------
            if Am_weight==1:
                    pass;
            else:
                    LM_local_scores,scores_list,present_label,scores=LM_model.prediction_from_trained_model(ys,scores_list)
                    COMB_AM_MT_local_scores = Am_weight * COMB_AM_MT_local_scores + (1-Am_weight) * LM_local_scores
            #---------------------------------------------------------------------

            #------------------------------------------------------------------------------------------------------------------------
            ys, score_1 = self.prediction_from_trained_model_beam_Search(i, ys, score_1, COMB_AM_MT_local_scores, beam, hyps, gamma, batch_size)
            ##---------------------------------------------------
            score_1,store_ended_hyps,store_ended_LLR=self.get_multiple_hypothesis(store_ended_hyps,store_ended_LLR,ys,score_1,i,maxlen)
            #----------------------------------------------------
             
            #print(score_1.shape)    
            #### removing blank predictions :::::::> ####  prdicting eos at the first token
            ##.pop(index) in python return a value   --->  careful
            
            #if not self.MT_flag:
            remove_blank_predictions_index=[index for index,element in enumerate(store_ended_hyps) if (len(element)==2 and element[0]==self.sos_id and element[1]==self.eos_id)==True]
            [store_ended_hyps.pop(element) for element in remove_blank_predictions_index]
            [store_ended_LLR.pop(element) for element in remove_blank_predictions_index]
            #----------------------------------------------------
            if self.MT_flag:
                    if len(store_ended_hyps)>=hyps*args.MT_input_beam:
                        break;
            else:
                    if len(store_ended_hyps)>=hyps:
                        break;
        if not self.MT_flag:
                #----------------------------------------------------
                ys=nn.utils.rnn.pad_sequence(store_ended_hyps,batch_first=True,padding_value=self.eos_id)
                score_1=nn.utils.rnn.pad_sequence(store_ended_LLR,batch_first=True,padding_value=np.log(len_bonus)) 
                #producing the correct_order
                #----------------------------------------------------
                # XS=[torch.sum(i) for i in store_ended_LLR]
                # XS1=sorted(((e,i) for i,e in enumerate(XS)),reverse=True)
                # correct_sorted_order=[i[1] for i in XS1]
                _,correct_sorted_order = torch.sort(torch.sum(score_1,dim=1),descending=True)
                #----------------------------------------------------
                ys=ys[correct_sorted_order]
                score_1=score_1[correct_sorted_order]
        else:
                ys=nn.utils.rnn.pad_sequence(store_ended_hyps,batch_first=True,padding_value=self.eos_id)
                score_1=nn.utils.rnn.pad_sequence(store_ended_LLR,batch_first=True,padding_value=np.log(len_bonus))
                #producing the correct_order
                #----------------------------------------------------
                _,correct_sorted_order = torch.sort(torch.sum(score_1,dim=1),descending=True)
                #----------------------------------------------------
                ys=ys[correct_sorted_order]
                score_1=score_1[correct_sorted_order]


        #breakpoint()
        if (self.MT_flag==False):
            ###for the botlenecks
            #breakpoint()
            ys=ys[:hyps]
            score_1=score_1[:hyps]
            
            ys_bneck=ys
            ys_bneck[ys_bneck == self.eos_id] = self.IGNORE_ID
            A1111,A2222,A333,dec_output_Bneck=self.prediction_from_trained_model(ys,rep_encoder_outputs,scores_list)
            ##------------------------------
            print(ys,torch.sum(score_1,dim=1))    
            #--------------------------------
        else:
            dec_output_Bneck=None
        
        #breakpoint()
        output_dict = {'ys':ys, 'score_1':score_1, 'dec_output':dec_output_Bneck}
        return output_dict
    ####################################
    ######################################################################
    ######################################################################
    #--------------------------------------------
    ######################################################################    
    ######################################################################
    # def get_charecters_for_sequences(self,input_tensor,Word_model,pad_index,eos_id,word_unk):
    #     """ Takes pytorch tensors as in put and print the text charecters as ouput,  
    #     replaces sos and eos as unknown symbols and ?? and later deletes them from the output string 
    #     """
    #     breakpoint()

    #     output_text_seq=[]
    #     final_token_seq=input_tensor.data.numpy()
    #     final_token_seq=np.where(final_token_seq>=pad_index,word_unk,final_token_seq)
    #     text_sym_sil_tok=Word_model.DecodeIds([word_unk])

    #     for i in final_token_seq:
    #         i=i.astype(np.int).tolist()
    #         text_as_string=Word_model.DecodeIds(i)
    #         text_as_string=text_as_string.replace(text_sym_sil_tok,"")
    #         output_text_seq.append(text_as_string)
    #     #breakpoint()
    #     return output_text_seq
    def get_charecters_for_sequences(self,input_tensor,Word_model,pad_index,eos_id,word_unk):
            ##made it little geniric may be required for future i dont know...?
            """ Takes pytorch tensors as in put and print the text charecters as ouput,  
            replaces sos and eos as unknown symbols and ?? and later deletes them from the output string"""

            #breakpoint()
            output_text_seq=[]
            if input_tensor.dim()==1:
                input_tensor = input_tensor.unsqueeze(0)


            final_token_seq=input_tensor.data.numpy()          
            text_sym_sil_tok=Word_model.DecodeIds([word_unk])

            for i in final_token_seq:
                ##delete the elements if the model produces good labels after the eos they need to be removed
                i=i.astype(np.int).tolist()
                utt_eos_index = i.index(eos_id) if eos_id in i else None

                #--------------------------------------
                ####if eos exists in the utterance remove the seq after eos
                if utt_eos_index:
                    i=i[:utt_eos_index]

                ##if no eos in the seq then take full seq
                i=[x for x in i if x<pad_index]
                #--------------------------------------

                text_as_string = Word_model.DecodeIds(i)
                #text_as_string = text_as_string.replace(text_sym_sil_tok," <UNK> ")
                output_text_seq.append(text_as_string)
            return output_text_seq



        #=======================================================================================
    #=======================================================================================
#==========================================================================

def cal_performance(pred, gold,IGNORE_ID,normalize_length=False,smoothing=0.0):
    """Calculate cross entropy loss, apply label smoothing if needed.
    Args:  pred: N x T x C, score before softmax ;;;; gold: N x T """

    pred = pred.view(-1, pred.size(2))
    gold = gold.contiguous().view(-1)

    loss = cal_loss(pred, gold,IGNORE_ID,normalize_length,smoothing)
    pred = pred.max(1)[1]
    non_pad_mask = gold.ne(IGNORE_ID)

    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()
    n_correct=n_correct/float(non_pad_mask.sum())

    n_correct=1.0-n_correct
    return loss, n_correct
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
#===============================================
#===============================================
def cal_loss(pred, gold,IGNORE_ID,normalize_length,smoothing):
    """Calculate cross entropy loss, apply label smoothing if needed.  """
    normalize_length=True
    if smoothing > 0.0:
        eps = smoothing
        n_class = pred.size(1)

        # Generate one-hot matrix: N x C.
        # Only label position is 1 and all other positions are 0
        # gold include -1 value (IGNORE_ID) and this will lead to assert error
        gold_for_scatter = gold.ne(IGNORE_ID).long() * gold
        one_hot = torch.zeros_like(pred).scatter(1, gold_for_scatter.view(-1, 1), 1)

        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / n_class

        log_prb = F.log_softmax(pred, dim=1)
        non_pad_mask = gold.ne(IGNORE_ID)
        n_word = non_pad_mask.sum().item()
        loss = -(one_hot * log_prb).sum(dim=1)   

        loss = loss.masked_select(non_pad_mask).sum() / n_word

    else:
        loss = F.cross_entropy(pred, gold,
                               ignore_index=IGNORE_ID,
                               reduction='elementwise_mean')
    return loss



