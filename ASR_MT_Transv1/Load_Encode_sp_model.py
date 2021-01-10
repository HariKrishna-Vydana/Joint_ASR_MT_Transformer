#!/usr/bin/python

import sys
import os
from os.path import join, isdir
import sentencepiece as spm

#--------------------------
def Load_Encode_sp_model(PATH,text):
        PATH_model = spm.SentencePieceProcessor()
        PATH_model.Load(join(PATH))

        #------------------------------------
        if '__word' in PATH:
                ##to account for two successive OOV labels
                utt_index=[]
                for word in text.split(' '):
                        utt_index+=PATH_model.EncodeAsIds(word)
        else:
                text_tokens = PATH_model.EncodeAsIds(text)
        #-----------------------------------------
        return text_tokens
#--------------------------

