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
from multiprocessing import Process
import random
import glob
import concurrent.futures


import sys
sys.path.insert(0, '/mnt/matylda3/vydana/HOW2_EXP/Joint_ASR_MT_Transformer')
from ASR_MT_Transv1.CMVN import CMVN
from ASR_MT_Transv1.Load_sp_model import Load_sp_models
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

        self.datareader=DataLoader.process_everything_in_parllel_dummy2
        #self.__load_data()

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
    #---------------------------------------
    #------------------------------------------

    def __load_data(self):
        print('inside __load_data.....')
        ###initilize the lists
        while True:
            self.__reset_the_data_holders()
            max_batch_label_len = self.max_batch_label_len
            random.shuffle(self.files)
            ###
            for inp_file in self.files:
                print(inp_file)
                ##############################################################################
                #########    Parllel data reading  ############################################
                with open(inp_file) as f:
                            with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
                                results=[executor.submit(self.datareader,line) for line in f]
                                for R in concurrent.futures.as_completed(results):

                                    #breakpoint()

                                    dataread_dict=R.result()

                                    if dataread_dict==None:
                                        continue;
                                   
                                    #--------------------------
                                    key = dataread_dict['key'];
                                    #print('************************',key)
                                    mat = dataread_dict['mat'];
                                    
                                    Src_tokens = dataread_dict['Src_tokens'];
                                    
                                    Src_Words_Text = dataread_dict['Src_Words_Text'];
                                    
                                    Tgt_tokens = dataread_dict['Tgt_tokens'];
                
                                    Tgt_Words_Text = dataread_dict['Tgt_Words_Text'];
        
                                    scp_path = dataread_dict['scp_path'];
                                    
                                    dataread_dict={}
                                    R._result = None
                                    ##############################################################################
                                    #########    Parllel data reading  finished and continueing old style ###########################################################
                                    ##############

                                    if (scp_path!='None'):
                                        ##ASR data-----
                                        if self.apply_cmvn:
                                            mat = CMVN(mat)

                                        ####pruning the Acoustic features based on length ###for joint model
                                        if (mat.shape[0]>self.max_feat_len) or (len(Src_tokens) > self.max_label_len):
                                            continue;

                                    else:   
                                        ####For MT Data
                                        ###Src_tokens more than self.max_feat_len or Tgt_tokens more than self.max_label_len should be  removed
                                        if (len(Src_tokens) > self.max_feat_len) or (len(Tgt_tokens) > self.max_label_len):
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
                                    #print(len(self.batch_Src_data), self.max_batch_len,total_labels_in_batch)
                                    if total_labels_in_batch > self.max_batch_label_len or len(self.batch_Src_data)==self.max_batch_len:
                                                # #==============================================================
                                                # ####to clumsy -------> for secound level of randomization 
                                                # CCCC=list(zip(batch_data,batch_names,batch_labels,batch_Tgt_Words_Text,batch_word_text,batch_label_length,batch_length,batch_Tgt_label_length,batch_word_text_length))
                                                # random.shuffle(CCCC)
                                                # batch_data,batch_names,batch_labels,batch_Tgt_Words_Text,batch_word_text,batch_label_length,batch_length,batch_Tgt_label_length,batch_word_text_length=zip(*CCCC)
                                                # #==============================================================

                                                print('making_batch.........................',self.queue.qsize())
                                                batch_data_dict = self.make_batching_dict()
                                                self.queue.put(batch_data_dict)
                                                ###after pushing data to lists reset them
                                                self.__reset_the_data_holders()

    def next(self, timeout=30000):
        #print('inside next......')
        #breakpoint()
        return self.queue.get(block=True, timeout=timeout)    

    @staticmethod 
    def process_everything_in_parllel_dummy2(line):
        #output_dict={'key':None,'mat':None,'Src_tokens':None,'Src_Words_Text':None,'Tgt_tokens':None,'Tgt_Words_Text':None,'scp_path':None}
        #============================
        split_lines=line.split(' @@@@ ')
        #============================
        key = split_lines[0]
        scp_path = split_lines[1] #will be 'None' fo MT setup
        scp_path = 'None' if scp_path == '' else scp_path
        #print('-------',key)
        #============================
        ### Char labels
        #============================
        src_text = split_lines[3] 
        src_tok = split_lines[4] 

        tgt_text = split_lines[5]
        tgt_tok = split_lines[6]

        # Src_tokens = src_tok
        # Tgt_tokens = tgt_tok

        if tgt_text=='None':
            ###ASR Data
            Tgt_tokens=list(map(lambda x: x*0, tgt_tok))

        if ('None' not in src_tok) and ('None' not in tgt_tok) and (len(src_tok)>0) and (len(tgt_tok)>0):

            Src_tokens = [int(i) for i in src_tok.split(' ')]
            Tgt_tokens = [int(i) for i in tgt_tok.split(' ')]
            #============================
            ### text 
            #============================
            Src_Words_Text = src_text.split(' ')
            Tgt_Words_Text = tgt_text.split(' ')
            #--------------------------                                
            #--------------------------
            if not (scp_path == 'None'):
                mat = kaldi_io.read_mat(scp_path)
            else:
                mat=np.zeros((100,83),dtype=np.float32)

            ###########################################
            output_dict={'key':key,
                        'mat':mat,
                        'Src_tokens':Src_tokens,
                        'Src_Words_Text':Src_Words_Text,
                        'Tgt_tokens':Tgt_tokens,
                        'Tgt_Words_Text':Tgt_Words_Text,
                        'scp_path':scp_path}
        else:
            return None

        return output_dict

#===================================================================

