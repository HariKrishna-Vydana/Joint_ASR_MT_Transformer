#!/usr/bin/python

import sys
import os
import torch
#----------------------------------------
#----------------------------------------
def forword_and_update(smp_no, trainflag, model, optimizer, input, smp_Src_labels, smp_Tgt_labels, smp_Src_labels_tc, accm_grad, clip_grad_norm):

        Decoder_out_dict = model(input, smp_Src_labels, smp_Tgt_labels,smp_Src_labels_tc)

        
        cost=Decoder_out_dict.get('cost')
        #=====================================
        if trainflag:
            cost=cost/accm_grad
            cost.backward()

            if clip_grad_norm != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            cost.detach() 
            ###gradient accumilation
            if(smp_no%accm_grad)==0:
                optimizer.step()
                optimizer.zero_grad()

        #--------------------------------------
        cost_cpu = cost.item()
        return Decoder_out_dict, cost_cpu
#----------------------------------------
#----------------------------------------

#----------------------------------------
#----------------------------------------
#---------------------------------------
def train_val_model(**kwargs):

        #breakpoint()

        smp_no=kwargs.get('smp_no')
        args = kwargs.get('args')
        model = kwargs.get('model')
        optimizer= kwargs.get('optimizer')
 
        trainflag = kwargs.get('trainflag')
        
        B1 = kwargs.get('data_dict')
        
        smp_Src_data = B1.get('smp_Src_data')
        smp_Src_labels = B1.get('smp_Src_labels')
        smp_Tgt_labels = B1.get('smp_Tgt_labels')
        
        ###for future
        smp_Src_Text = B1.get('smp_Src_Text') 
        smp_Tgt_Text = B1.get('smp_Tgt_Text')  

        smp_Src_labels_tc = B1.get('smp_Src_labels_tc') 
        smp_Src_Text_tc = B1.get('smp_Src_Text_tc')
       
        #################finished expanding the keyword arguments#########
        ##===========================================
        #============================================
        ###################################################################
        input = torch.from_numpy(smp_Src_data).float()

        smp_Src_labels = torch.LongTensor(smp_Src_labels)
        smp_Tgt_labels = torch.LongTensor(smp_Tgt_labels)
        smp_Src_labels_tc = torch.LongTensor(smp_Src_labels_tc)

        #-----------------------------------------------------------------
        input = input.cuda() if args.gpu else input        
        smp_Src_labels = smp_Src_labels.cuda() if args.gpu else smp_Src_labels
        smp_Tgt_labels = smp_Tgt_labels.cuda() if args.gpu else smp_Tgt_labels

        smp_Src_labels_tc = smp_Src_labels_tc.cuda() if args.gpu else smp_Src_labels_tc        
        #--------------------------------

        OOM=False
        if trainflag:
            try:
                Decoder_out_dict, cost_cpu = forword_and_update(smp_no, trainflag, model, optimizer, input, smp_Src_labels, smp_Tgt_labels, smp_Src_labels_tc, args.accm_grad, args.clip_grad_norm)

            except Exception as e:
               
                if 'CUDA out of memory' in str(e):
                  OOM=True
                  torch.cuda.empty_cache()
                  print("The model in OOM condition", "smp_no", smp_no, "batch size for the batch is:", smp_Src_labels.shape, smp_Tgt_labels.shape)
                  #break;
                else:
                    ####print if some other error occurs
                    print("There is some",str(e))



            ###When there is oom eror make the batch size 2
            if OOM:
                input=input[:2]
                batch_size = smp_Src_labels.shape[0]
                smp_Src_labels = smp_Src_labels[:2]
                smp_Tgt_labels = smp_Tgt_labels[:2]
                smp_Src_labels_tc = smp_Src_labels_tc[:2]

                print("The model running under OOM condition","smp_no",smp_no,"batch size for the batch is:",2)
                Decoder_out_dict, cost_cpu = forword_and_update(smp_no, trainflag, model, optimizer, input, smp_Src_labels, smp_Tgt_labels, smp_Src_labels_tc, args.accm_grad, args.clip_grad_norm)

        else:
            with torch.no_grad():
                    Decoder_out_dict, cost_cpu = forword_and_update(smp_no, trainflag, model, optimizer, input, smp_Src_labels, smp_Tgt_labels,smp_Src_labels_tc,args.accm_grad, args.clip_grad_norm)
        #--------------------------------        
        #print(Decoder_out_dict.keys(), cost_cpu)
        ###output a dict
        #==================================================    
        Output_trainval_dict={
                            'cost_cpu': cost_cpu,
                            'dec_slf_attn_list': Decoder_out_dict.get('dec_slf_attn_list'),
                            'dec_enc_attn_list': Decoder_out_dict.get('dec_enc_attn_list'),
                            'Char_cer': Decoder_out_dict.get('Char_cer'),
                            'Word_cer': Decoder_out_dict.get('Word_cer')}
        return Output_trainval_dict
#=========================================================
