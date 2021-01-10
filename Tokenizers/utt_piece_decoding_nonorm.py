#!/usr/bin/python

import sys
import os
from os.path import join, isdir

import sentencepiece as spm
sp = spm.SentencePieceProcessor()

#--------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------

model_prev=sys.argv[1]
text_file=sys.argv[2]
outpath=sys.argv[3]
name_suffix=sys.argv[4]

model_list=[str(model_prev+'_bpe'),str(model_prev+'_char'),str(model_prev+'_unigram'),str(model_prev+'_word')]


#model_list=[str(model_prev+'_bpe'), str(model_prev+'_char')]


for model in model_list:
        
	
        sp.Load(join(model+".model"))
        bname=os.path.basename(model)
        outname=join(outpath,bname+'__'+str(text_file)+'_decoded'+str(name_suffix))

        out=open(outname,'w+')

        f=open(str(text_file),'r')
        f=f.readlines()
        
        for line in f:
                #print(line)
                name=line.split(' ')[0]
                line=" ".join(line.split(' ')[1:])

                #------------------------------------------------
                # if '_word' in model:
                #         line1=line.split(' ')
                #         utt_index=[]
                        for word in line1:
                             utt_index.append(sp.EncodeAsIds(word))   
                # else:
                #         utt_index=sp.EncodeAsIds(line)
                #------------------------------------------------

                utt_index=sp.EncodeAsIds(line)



                #------------------------------------------------
                utt_index_str=[str(j) for j in utt_index ]
                #------------------------------------------------
                out.write(name+' '+" ".join(utt_index_str)+"\n")
		#------------------------------------------------


