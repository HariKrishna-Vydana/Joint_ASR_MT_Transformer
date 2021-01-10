#!/usr/bin/python

import numpy as np


import sys
import sacrebleu
from sacrebleu import sentence_bleu,corpus_bleu

SMOOTH_VALUE_DEFAULT=1e-8
#sentence_bleu(hypothesis: str,references: List[str],smooth_method: str = 'floor',smooth_value: float = SMOOTH_VALUE_DEFAULT,use_effective_order: bool = True)
#reference = [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
#candidate = ['this', 'is', 'a', 'test']



ref_file=str(sys.argv[1])
hyp_file=str(sys.argv[2])
print(ref_file,hyp_file)

#----------------------------------------------------------------------------------------------
f = open(str(ref_file),'r')
ref_dict = {line.strip().split(' ')[0]:line.strip().split(' ')[1:] for line in f.readlines()}


ref_dict_list=[]
for key in list(ref_dict.keys()):
        ref_utt=ref_dict.get(key," ")
        if len(ref_utt)>1:
                ref_dict_list.append(" ".join(ref_utt))
        else:
                ref_dict_list.append(ref_utt[0])

g=open(str(hyp_file),'r')
hyp_dict={line.strip().split(' ')[0]:line.strip().split(' ')[1:] for line in g.readlines()}
#-----------------------------------------------------------------------

hyp_dict_list=[]
for key in list(ref_dict.keys()):
        hyp_utt=hyp_dict.get(key," ")

        if len(hyp_utt)>1:
                hyp_dict_list.append(" ".join(hyp_utt))
        else:
                hyp_utt = hyp_utt[0] if hyp_utt else " "
                hyp_dict_list.append(hyp_utt[0])
#-----------------------------------------------------------------------

#print(hyp_dict_list, ref_dict_list)
#----------------------------------------------------------------------------------------------
Bleu_score=corpus_bleu(hyp_dict_list,[ref_dict_list],smooth_value=SMOOTH_VALUE_DEFAULT,smooth_method='exp',use_effective_order='True')
print('BLUE:===>',Bleu_score.score)

#---------------------------------------------------------------------------------------------

