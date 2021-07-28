#! usr/bin/bash




#model_dir="/mnt/matylda3/vydana/HOW2_EXP/IWSLT2021_JointSLT_systems/models/Transformer_Mustc_enc12_256_1024_dec6_256_1024_enc6_256_1024_dec6_256_1024_3000_accm8"
#model_dir="/mnt/matylda3/vydana/HOW2_EXP/IWSLT2021_JointSLT_systems/models/Transformer_Mustc_enc12_512_2048_dec6_512_2048_enc6_512_2048_dec6_512_2048_4000_accm8_3datasets"
#model_dir="/mnt/matylda3/vydana/HOW2_EXP/IWSLT2021_Joint/models/ASR_MT_Transformer_enc12_1024_4096_dec6_1024_4096_enc6_1024_4096_dec6_1024_4096_7000_accm2_lc.rm_tc"
#model_dir="/mnt/matylda3/vydana/HOW2_EXP/IWSLT2021_Joint/models/ASR_MT_Transformer_enc12_1024_4096_dec6_1024_4096_enc6_1024_4096_dec6_1024_4096_7000_accm2_lc.rm_tc"

#model_dir="/mnt/matylda3/vydana/HOW2_EXP/IWSLT2021_Joint/models/ASR_MT_Transformer_enc12_1024_4096_dec6_1024_4096_enc6_1024_4096_dec6_1024_4096_7000_accm2_lc.rm_tc_pretrained_weights"



#model_dir="/mnt/matylda3/vydana/HOW2_EXP/IWSLT2021_Joint/models/ASR_Libri_MT_Transformer_enc12_1024_4096_dec6_1024_4096_enc6_1024_4096_dec6_1024_4096_7000_accm2_tc.tok_tc_pretrained_weights"
#model_dir="/mnt/matylda3/vydana/HOW2_EXP/IWSLT2021_Joint/models/ASR_MT_Transformer_enc12_1024_4096_dec6_1024_4096_enc6_1024_4096_dec6_1024_4096_7000_accm2_tc.tok_tc_pretrained_weights"
#model_dir="/mnt/matylda3/vydana/HOW2_EXP/IWSLT2021_Joint/models/ASR_MT_Transformer_enc12_1024_4096_dec6_1024_4096_enc6_1024_4096_dec6_1024_4096_7000_accm2_tc.tok_tc_pretrained_weights_MTtraining"



#model_dir="/mnt/matylda3/vydana/HOW2_EXP/HOW2_IWSLT2021/HOW2_IWSLT2021_Joint/models/ASR_MT_Transformer_enc12_1024_4096_dec6_1024_4096_enc6_1024_4096_dec6_1024_4096_7000_accm2_tc.tok_tc_pretrained_weights_5K/"



model_dir="/mnt/matylda3/vydana/HOW2_EXP/HOW2_IWSLT2021/HOW2_IWSLT2021_Joint/models/ASR_MT_Transformer_enc12_1024_4096_dec6_1024_4096_enc6_1024_4096_dec6_1024_4096_7000_accm2_tc.tok_tc_pretrained_weights_3K/"
SWA_random_tag="$RANDOM"

est_cpts=8
ignore_cpts=0
python /mnt/matylda3/vydana/HOW2_EXP/Joint_ASR_MT_Transformer/ASR_MT_Transv1/Get_SWA_weights.py $model_dir $SWA_random_tag $est_cpts $ignore_cpts

