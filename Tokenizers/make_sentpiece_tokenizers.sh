#!/usr/bin/bash

text_file=full_text_id.en

#model="models_PTG/PTG_Tok_"


model="models_ENG/ENG_Tok_"
output_path="models/"
name_suffix='.txt'


#-----------
cut -d " " -f1 $text_file>utt_id
#-----------
cut -d " " -f2- $text_file|sed 's/  */ /g'>utt_text
#-----------

#Special_string="" 
user_generated_symbols="<HES>,<UNK>,<BOS>,<EOS>,{LIPSMACK},{BREATH},{LAUGH},{COUGH},',-,.,?,!,:,;,COMMA"

cat utt_text | sed 's/,/COMMA/g' >utt_text_usersymbols

paste -d " " utt_id utt_text_usersymbols > "$text_file"_normalized


no_of_tokens=5000
mkdir -pv $model

python utt_piece_training_nonorm.py utt_text_usersymbols $model $no_of_tokens $user_generated_symbols









