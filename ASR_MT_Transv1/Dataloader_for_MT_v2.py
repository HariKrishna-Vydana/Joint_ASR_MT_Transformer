#!/usr/bin/python
import kaldi_io
import sys
import os
from os.path import join, isdir
from numpy.random import permutation
import itertools
import keras
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import queue
from threading  import Thread
import random
import glob

import sys
sys.path.insert(0, '/mnt/matylda3/vydana/HOW2_EXP/MT_Transformer/MT_TransV1')
from CMVN import CMVN
from Load_sp_model import Load_sp_models

#===============================================
#-----------------------------------------------  
class DataLoader(object):

    def __init__(self,files, max_batch_label_len, max_batch_len, max_feat_len, max_label_len, Src_model, Tgt_model, queue_size=100,apply_cmvn=1):

        self.files = files ####
        if self.files==[]:
                print('input to data generator in empty')
                exit(0)

        self.Src_model = Src_model
        self.Tgt_model = Tgt_model
        self.max_batch_len = max_batch_len
        self.max_batch_label_len = max_batch_label_len
        self.max_feat_len = max_feat_len
        self.max_label_len = max_label_len
        self.apply_cmvn = apply_cmvn


        self.queue = queue.Queue(queue_size)
        self.Src_padding_id = self.Src_model.__len__()
        self.Tgt_padding_id = self.Tgt_model.__len__()
        self.word_space_token   = self.Src_model.EncodeAsIds('_____')[0]
        
    
        self._thread = Thread(target=self.__load_data)
        self._thread.daemon = True
        self._thread.start()

    
    def __reset_the_data_holders(self):


        self.batch_names=[]
        self.batch_Src_data=[]
        self.batch_Src_length=[]

        self.batch_Src_labels=[]      
        self.batch_Src_label_length=[]
        
        self.batch_Src_text=[]
        self.batch_Src_text_length=[]

        self.batch_Tgt_labels=[]
        self.batch_Tgt_label_length=[]

        self.batch_Tgt_text=[]
        self.batch_Tgt_text_length=[]
    
    #---------------------------------------------------------------------
    def make_batching_dict(self):
       
        #----------------------------------------
        smp_Src_data= pad_sequences(self.batch_Src_data,maxlen=max(self.batch_Src_length),dtype='float32',padding='post',value=0.0)
        smp_Src_labels = pad_sequences(self.batch_Src_labels,maxlen=max(self.batch_Src_label_length),dtype='int32',padding='post',value=self.Src_padding_id) 
        smp_Tgt_labels = pad_sequences(self.batch_Tgt_labels,maxlen=max(self.batch_Tgt_label_length),dtype='int32',padding='post',value=self.Tgt_padding_id)
        
        smp_Src_Text = pad_sequences(self.batch_Src_text, maxlen=max(self.batch_Src_text_length),dtype=object,padding='post',value='')
        smp_Tgt_Text = pad_sequences(self.batch_Tgt_text, maxlen=max(self.batch_Tgt_text_length),dtype=object,padding='post',value='')

        batch_data_dict={
            'smp_names':self.batch_names,
            'smp_Src_data':smp_Src_data,
            'smp_Src_labels':smp_Src_labels,
            'smp_Tgt_labels':smp_Tgt_labels,
            'smp_Src_Text':smp_Src_Text,
            'smp_Tgt_Text':smp_Tgt_Text,
            'smp_Src_data_length':self.batch_Src_length,
            'smp_Src_label_length':self.batch_Src_label_length,
            'smp_Src_text_length':self.batch_Src_text_length,
            'smp_Tgt_label_length':self.batch_Tgt_label_length,
            'smp_Tgt_text_length':self.batch_Tgt_text_length}
        return batch_data_dict
    #------------------------------------------
    #------------------------------------------
    def __load_data(self):
        ###initilize the lists
        while True:
            self.__reset_the_data_holders()
            max_batch_label_len = self.max_batch_label_len
            random.shuffle(self.files)

            for inp_file in self.files:
                with open(inp_file) as f:
                    for line in f:
                        #============================
                        split_lines=line.split(' @@@@ ')
                        #============================
                        ####this is mostly for the joint model 
                        ###usuvally MT setup will not have scp so just fill the space with the default vallues

                        #breakpoint()
                        ##assigining
                        key = split_lines[0]
                        scp_path = split_lines[1] #will be 'None' fo MT setup
                        scp_path = 'None' if scp_path == '' else scp_path
                        #============================
                        ### Char labels
                        #============================

                        src_text = split_lines[3] 
                        src_tok = split_lines[4] 

                        if len(src_tok)>0:
                            src_tok = [int(i) for i in src_tok.split(' ')]
                        else:
                                continue;                        
                        #============================
                        ##Word models
                        #============================
                        tgt_text = split_lines[5]
                        tgt_tok = split_lines[6]
                       
                         
                        if len(tgt_tok)>0:
                            tgt_tok = [int(i) for i in tgt_tok.split(' ')]
                        else:
                                continue;
                        #============================
                        ### text 
                        #============================
                        Src_tokens = src_tok
                        Tgt_tokens = tgt_tok

                        Src_Words_Text = src_text.split(' ')
                        Tgt_Words_Text = tgt_text.split(' ')
                        #--------------------------
                        if not (scp_path == 'None'):
                            #breakpoint()
                            mat = kaldi_io.read_mat(scp_path)
                             
                            if self.apply_cmvn:
                                mat = CMVN(mat)
                                
                            ####pruning the Acoustic features based on length ###for joint model
                            if (mat.shape[0]>self.max_feat_len) or (len(Src_tokens) > self.max_label_len):
                                #print("key,mat.shape,Src_Words_Text,Src_tokens,self.max_label_len",key,mat.shape,len(Src_Words_Text),len(Src_tokens),self.max_label_len)
                                continue;
                        else:
                            mat=np.zeros((100,249),dtype=np.float32)
                            ####For MT model 
                            ###Src_tokens more than self.max_feat_len or Tgt_tokens more than self.max_label_len
                            ### should be  removed
                            ###
                            if (len(Src_tokens) > self.max_feat_len) or (len(Tgt_tokens) > self.max_label_len):
                                #print("key,Src_tokens, self.max_feat_len, Tgt_tokens, self.max_label_len",key,len(Src_tokens), self.max_feat_len, len(Tgt_tokens), self.max_label_len)
                                continue;

                        #--------------------------
                        #==============================================================
                        ###Add to the list
                        ####
                        self.batch_Src_data.append(mat)                
                        self.batch_names.append(key)
                        self.batch_Src_length.append(mat.shape[0])

                        self.batch_Src_labels.append(Src_tokens)
                        self.batch_Src_label_length.append(len(Src_tokens))
                        
                        self.batch_Tgt_labels.append(Tgt_tokens)
                        self.batch_Tgt_label_length.append(len(Tgt_tokens))

                        self.batch_Src_text.append(Src_Words_Text)
                        self.batch_Src_text_length.append(len(Src_Words_Text))

                        self.batch_Tgt_text.append(Tgt_Words_Text)
                        self.batch_Tgt_text_length.append(len(Tgt_Words_Text))   
                        #==============================================================
                        #==============================================================
                        # total_labels_in_batch is used to keep track of the length of sequences in a batch, just make sure it does not overflow the gpu
                        ##in general lstm training we are not using this because self.max_batch_len will be around 10-20 and self.max_batch_label_len is usuvally set very high     
                        #-------------------------------------------------------------------------------
                        if (scp_path != 'None'):
                            expect_len_of_features=max(max(self.batch_Src_length,default=0)/4,mat.shape[0]/4)
                            expect_len_of_labels=max(max(self.batch_Tgt_label_length,default=0),len(Tgt_tokens))
                            total_labels_in_batch= (expect_len_of_features + expect_len_of_labels)*(len(self.batch_names)+4)
                            total_labels_in_batch = int(total_labels_in_batch)
                        else:
                            expect_len_of_features=max(max(self.batch_Src_label_length,default=0),len(Src_tokens))
                            expect_len_of_labels=max(max(self.batch_Tgt_label_length,default=0),len(Tgt_tokens))
                            total_labels_in_batch= (expect_len_of_features + expect_len_of_labels)*(len(self.batch_names)+4)
                        #-------------------------------------------------------------------------------

                        ###check if ypu have enough labels output and if you have then push to the queue
                        ###else keep adding them to the lists
                        #print(len(self.batch_Src_data), self.max_batch_len)
                        if total_labels_in_batch > self.max_batch_label_len or len(self.batch_Src_data)==self.max_batch_len:
                                    # #==============================================================
                                    # ####to clumsy -------> for secound level of randomization 
                                    # CCCC=list(zip(batch_data,batch_names,batch_labels,batch_Tgt_Words_Text,batch_word_text,batch_label_length,batch_length,batch_Tgt_label_length,batch_word_text_length))
                                    # random.shuffle(CCCC)
                                    # batch_data,batch_names,batch_labels,batch_Tgt_Words_Text,batch_word_text,batch_label_length,batch_length,batch_Tgt_label_length,batch_word_text_length=zip(*CCCC)
                                    # #==============================================================

                                    batch_data_dict = self.make_batching_dict()
                                    self.queue.put(batch_data_dict)
                                    ###after pushing data to lists reset them
                                    self.__reset_the_data_holders()
            

            if len(self.batch_names)>0:
                ### Collect the left over stuff  as the last batch
                #-----------------------------------------------
                batch_data_dict = self.make_batching_dict()
                self.queue.put(batch_data_dict)

    def next(self, timeout=30000):
        return self.queue.get(block=True, timeout=timeout)
#===================================================================


# sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/KAT_Attention')
# import Attention_arg
# from Attention_arg import parser
# args = parser.parse_args()
# print(args)


# ###debugger
# args.Src_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/models/Timit_PHSEQ_100/Timit_PHSEQ__100__word.model'
# args.Tgt_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/models/Timit_PHSEQ_100/Timit_PHSEQ__100__word.model'
# args.text_file = '/mnt/matylda3/vydana/benchmarking_datasets/Timit/All_text'
# args.train_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/scp_files/train/'
# args.dev_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/scp_files/dev/'
# Src_model=Load_sp_models(args.Src_model_path)
# Tgt_model=Load_sp_models(args.Tgt_model_path)
# train_gen = DataLoader(files=glob.glob(args.train_path + "*"),max_batch_label_len=20000, max_batch_len=4,max_feat_len=2000,max_label_len=200,Src_model=Src_model,Tgt_model=Tgt_model,text_file=args.text_file)
# for i in range(10):
#     B1 = train_gen.next()
#     print(B1.keys())
#     #breakpoint()

