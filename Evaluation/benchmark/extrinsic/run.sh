NUM_GPU=1
export CUDA_VISIBLE_DEVICES=0,

CACHE_DIR=.cache

python train.py \
    --model_name_or_path /content/drive/MyDrive/학부연구생_2/model_outputs2/Global_FirstCL_data2_st630_lr45_bs32_temp005_sd174/training_step_630_train_mle_loss_0.0_train_cl_loss_0.021_dev_ppl_608192.769\
    --cache_dir $CACHE_DIR \
    --ckpt_dir nli-results-2/Global_bias_mitigation



