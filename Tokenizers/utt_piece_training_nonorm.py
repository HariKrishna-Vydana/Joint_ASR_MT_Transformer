#!/usr/bin/python

import sys
import os
from os.path import join, isdir

import sentencepiece as spm
sp = spm.SentencePieceProcessor()
#s.Load('spm.model')

utt_text_normalized=sys.argv[1]
model_prefix=sys.argv[2]
vocab_size=sys.argv[3]

user_generated_strings = sys.argv[4]


###replaced , with a #  just to use comma as a special symbol

#user_generated_strings="<HES>,<UNK>,<BOS>,<EOS>,{LIPSMACK},{BREATH},{LAUGH},{COUGH},',-,.,?,!,:,;COMMA"
#control_symbols=","

#control_symbols = user_generated_strings
#user_generated_strings = ","



#user_generated_strings = user_generated_strings + control_symbols
print(user_generated_strings.split(','))



#--------------------------------------------------------------------------------------------------------------------------------------------------------
unigram_str='--input='+str(utt_text_normalized)+' --model_prefix='+str(model_prefix)+'_unigram'+' --vocab_size='+str(vocab_size)+' --model_type=unigram --character_coverage=1.0 --unk_id=1 --bos_id=-1 --eos_id=-1 --pad_id=0 --pad_piece=<blk> --user_defined_symbols='+str(user_generated_strings) 
#--------------------------------------------------------------------------------------------------------------------------------------------------------
bpe_str='--input='+str(utt_text_normalized)+' --model_prefix='+str(model_prefix)+'_bpe'+' --vocab_size='+str(vocab_size)+' --model_type=bpe --character_coverage=1.0 --unk_id=1 --bos_id=-1 --eos_id=-1 --pad_id=0 --pad_piece=<blk> --user_defined_symbols='+str(user_generated_strings)
#--------------------------------------------------------------------------------------------------------------------------------------------------------
word_str='--input='+str(utt_text_normalized)+' --model_prefix='+str(model_prefix)+'_word'+' --vocab_size='+str(vocab_size)+' --model_type=word --character_coverage=1.0 --unk_id=1 --bos_id=-1 --eos_id=-1 --pad_id=0 --pad_piece=<blk> --user_defined_symbols='+str(user_generated_strings)
#----------------------------------------------------------------------------------------------------------------------------------------------------------
char_str='--input='+str(utt_text_normalized)+' --model_prefix='+str(model_prefix)+'_char'+' --vocab_size='+str(vocab_size)+' --model_type=char --character_coverage=1.0 --unk_id=1 --bos_id=-1 --eos_id=-1 --pad_id=0 --pad_piece=<blk> --user_defined_symbols='+str(user_generated_strings)

#--------------------------------------------------------------------------------------------------------------------------------------------------------

print(unigram_str,bpe_str,word_str,char_str)

spm.SentencePieceTrainer.Train(unigram_str)

spm.SentencePieceTrainer.Train(bpe_str)

spm.SentencePieceTrainer.Train(word_str)

spm.SentencePieceTrainer.Train(char_str)


