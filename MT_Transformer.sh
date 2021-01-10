#! /bin/sh
#
#$ -q long.q@supergpu*,long.q@facegpu*,long.q@pc*,long.q@dellgpu*
#$ -l gpu=1,gpu_ram=7G,ram_free=7G,matylda3=0.5

#$ -o /mnt/matylda3/vydana/HOW2_EXP/MT_Transformer/log/Transformer_T5.log
#$ -e /mnt/matylda3/vydana/HOW2_EXP/MT_Transformer/log/Transformer_T5.log



PPATH="/mnt/matylda3/vydana/HOW2_EXP/MT_Transformer"
cd "$PPATH"
export PYTHONUNBUFFERED=TRUE

only_scoring='False'
scoring_path='/mnt/matylda3/vydana/HOW2_EXP/MT_Transformer/MT_Transformer'
stage=3


PPATH="/mnt/matylda3/vydana/HOW2_EXP/MT_Transformer"
gpu=1
max_batch_len=100
max_batch_label_len=8000
max_feat_len=300
max_label_len=300


tr_disp=50
vl_disp=10
validate_interval=1000
max_val_examples=2305

learning_rate=0.0001
early_stopping=0    
clip_grad_norm=5


input_size=249
hidden_size=256
kernel_size=3
stride=2
in_channels=1
out_channels=64
conv_dropout=0.3
isresidual=0
label_smoothing=0.1

####encoder parameters
encoder_layers=6
encoder_dmodel=256
encoder_heads=4
encoder_dinner=1024
encoder_dropout=0.1
encoder_ff_dropout=0.3

####decoder parameters
dec_embd_vec_size=256
decoder_dmodel=256
decoder_heads=4
decoder_dinner=1024
decoder_dropout=0.1
decoder_ff_dropout=0.3
decoder_layers=6
tie_dec_emb_weights=0

warmup_steps=5000

teacher_force=0.6
min_F_bands=5
max_F_bands=30
time_drop_max=2
time_window_max=1

weight_noise_flag=0
reduce_learning_rate_flag=0
spec_aug_flag=0


pre_trained_weight="0"

plot_fig_validation=0
plot_fig_training=0
start_decoding=0

#---------------------------
Src_model_path='/mnt/matylda3/vydana/HOW2_EXP/MT_Transformer/Tokenizers/models_ENG/ENG_Tok__bpe.model'
Tgt_model_path='/mnt/matylda3/vydana/HOW2_EXP/MT_Transformer/Tokenizers/models_PTG/PTG_Tok__bpe.model'

src_text_file='/mnt/matylda3/vydana/HOW2_EXP/MT_Transformer/Tokenizers/full_text_id.en_normalized'
tgt_text_file='/mnt/matylda3/vydana/HOW2_EXP/MT_Transformer/Tokenizers/full_text_id.pt_normalized'




###they contain utterance lists similar to the scp files
train_path='/mnt/matylda3/vydana/HOW2_EXP/MT_Transformer/scp_files/train/'
dev_path='/mnt/matylda3/vydana/HOW2_EXP/MT_Transformer/scp_files/dev/'
test_path='/mnt/matylda3/vydana/HOW2_EXP/MT_Transformer/scp_files/dev/'


data_dir="$PPATH/MT_data_files/"
mkdir -pv $data_dir

model_file="Transformer_T5"
model_dir="$PPATH/models/$model_file"
weight_text_file="$PPATH/weight_files/$model_file"
Res_text_file="$PPATH/weight_files/$model_file"_Res
mkdir -pv $model_dir

output_file="$PPATH/log/$model_file".log
log_file="$PPATH/log/$model_file".log

if [[ ! -w $weight_text_file ]]; then touch $weight_text_file; fi
if [[ ! -w $Res_text_file ]]; then touch $Res_text_file; fi
	
echo "$model_dir"
echo "$weight_file"
echo "$Res_file"


if [ $stage -le 1 ]; then
# #---------------------------------------------------------------------------------------------
##### making the data preperation for the experiment
stdbuf -o0  python /mnt/matylda3/vydana/HOW2_EXP/MT_Transformer/Make_training_scps_Transformer.py \
						--data_dir $data_dir \
                                                --src_text_file $src_text_file \
                                                --tgt_text_file $tgt_text_file \
 						--train_path $train_path \
 						--dev_path $dev_path \
 						--Tgt_model_path $Tgt_model_path \
 						--Src_model_path $Src_model_path
##---------------------------------------------------------------------------------------------
fi
###scp wrd char


#To avoid stale file handle error
random_tag="$RANDOM"_temp
new_data_dir="$data_dir""$random_tag"

mkdir -pv $new_data_dir
cp "$data_dir"*_scp "$new_data_dir"
data_dir="$new_data_dir"/
echo "$data_dir"
#exit 1



if [ $stage -le 2 ]; 
then

# #---------------------------------------------------------------------------------------------
stdbuf -o0  python /mnt/matylda3/vydana/HOW2_EXP/MT_Transformer/Transformer_MT_Training.py \
						--data_dir $data_dir \
 						--gpu $gpu \
 						--train_path $train_path \
 						--dev_path $dev_path \
 						--Tgt_model_path $Tgt_model_path \
 						--Src_model_path $Src_model_path \
 						--max_batch_label_len $max_batch_label_len \
 						--tr_disp $tr_disp \
 						--validate_interval $validate_interval \
 						--weight_text_file $weight_text_file \
 						--Res_text_file $Res_text_file \
 						--model_dir $model_dir \
 						--max_val_examples $max_val_examples \
 						--learning_rate $learning_rate \
 						--early_stopping $early_stopping \
 						--vl_disp $vl_disp \
 						--clip_grad_norm $clip_grad_norm \
 						--label_smoothing $label_smoothing \
 						--pre_trained_weight $pre_trained_weight \
 						--plot_fig_validation $plot_fig_validation \
 						--plot_fig_training $plot_fig_training \
 						--encoder_layers $encoder_layers \
						--encoder_dmodel $encoder_dmodel \
						--encoder_heads $encoder_heads \
						--encoder_dinner $encoder_dinner \
						--encoder_dropout $encoder_dropout \
						--encoder_ff_dropout $encoder_ff_dropout \
						--dec_embd_vec_size $dec_embd_vec_size \
						--decoder_dmodel $decoder_dmodel \
						--decoder_heads $decoder_heads \
						--decoder_dinner $decoder_dinner \
						--decoder_dropout $decoder_dropout \
						--decoder_ff_dropout $decoder_ff_dropout \
						--decoder_layers $decoder_layers \
						--tie_dec_emb_weights $tie_dec_emb_weights \
						--warmup_steps $warmup_steps \
						--max_batch_len $max_batch_len\
						--max_feat_len $max_feat_len \
						--max_label_len $max_label_len \
#---------------------------------------------------------------------------------------------
fi 


if [ $stage -le 3 ]; 
then


SWA_random_tag="$RANDOM"

gpu=0
#######this should have at maximum number of files to decode if you want to decode all the file then this should be length of lines in scps
max_jobs_to_decode=2400 
mem_req_decoding=10G
len_pen=1

for len_pen in 0.8 0.9 1.0 1.1 1.2 1.3 1.4 
do

for test_fol in $dev_path
do
D_path=${test_fol%*/}
D_path=${D_path##*/}
echo "$test_fol"
echo "$D_path"
for beam in 10 #1 2 3 4 5 6 7 8 9 10
do 
decoding_tag="_decoding_v1_beam_$beam""_$D_path""_len_pen_"$len_pen
log_path="$model_dir"/decoding_log_$decoding_tag
echo "$log_path"

mkdir -pv "$log_path"
mkdir -pv "$log_path/scoring"


if [ $only_scoring != 'True' ]; 
then


####looping just for the first job to do "Stocastic---Weight---Averaging" 
for max_jobs in 1 $max_jobs_to_decode 
do

/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1/utils/queue.pl \
	--max-jobs-run $max_jobs \
	-q short.q@@stable,short.q@@blade \
	--mem $mem_req_decoding \
	-l matylda3=0.01,ram_free=$mem_req_decoding,tmp_free=10G \
	JOB=1:$max_jobs \
	-l 'h=!blade063' \
	$log_path/decoding_job.JOB.log \
	python /mnt/matylda3/vydana/HOW2_EXP/MT_Transformer/Transformer_MT_decoding_v2.py \
	--gpu $gpu \
	--model_dir $model_dir \
	--Decoding_job_no JOB \
	--beam $beam \
	--dev_path $test_fol \
	--SWA_random_tag $SWA_random_tag \
        --len_pen $len_pen

done



cat $log_path/decoding_job*.log|grep "nbest"|awk -F ' = ' '{print $2, $3}'| tr -s " "> $log_path/scoring/hyp_val_file
cat $log_path/decoding_job*.log|grep "nbest"|awk -F ' = ' '{print $2, $4}'| tr -s " "> $log_path/scoring/ref_val_file


# cat $log_path/scoring/hyp_val_file | sed 's/⁇//g' > $log_path/scoring/hyp_val_file_fil

cat $log_path/scoring/hyp_val_file | sed 's/COMMA/,/g' > $log_path/scoring/hyp_val_file_fil
cat $log_path/scoring/ref_val_file | sed 's/COMMA/,/g' > $log_path/scoring/ref_val_file_fil


python $scoring_path/compute_scare_corpus_Blue.py $log_path/scoring/ref_val_file_fil $log_path/scoring/hyp_val_file_fil > $log_path/scoring/Bleu_score

cat $log_path/scoring/Bleu_score


else




cat $log_path/decoding_job*.log|grep "nbest"|awk -F ' = ' '{print $2, $3}'| tr -s " "> $log_path/scoring/hyp_val_file
cat $log_path/decoding_job*.log|grep "nbest"|awk -F ' = ' '{print $2, $4}'| tr -s " "> $log_path/scoring/ref_val_file

cat $log_path/scoring/hyp_val_file | sed 's/COMMA/,/g' > $log_path/scoring/hyp_val_file_fil
cat $log_path/scoring/ref_val_file | sed 's/COMMA/,/g' > $log_path/scoring/ref_val_file_fil


python $scoring_path/compute_scare_corpus_Blue.py $log_path/scoring/ref_val_file_fil $log_path/scoring/hyp_val_file_fil > $log_path/scoring/Bleu_score

cat $log_path/scoring/Bleu_score
fi


done
done
done

fi

###---------------------------------------------------------------------------------------------
