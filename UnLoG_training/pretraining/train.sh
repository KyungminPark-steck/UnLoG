CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python /content/drive/MyDrive/UnLoG/UnLoG_training/pretraining/train.py\
    --model_name bert-large-uncased\
    --train_path /content/drive/MyDrive/Bias_contrastive_learning/data/clean.csv\
    --dev_path /content/drive/MyDrive/Bias_contrastive_learning/data/wikitext103_raw_v1_validation.txt\
    --seqlen 64\
    --number_of_gpu 1\
    --batch_size_per_gpu 16\
    --gradient_accumulation_steps 2\
    --effective_batch_size 32\
    --total_steps 350\
    --print_every 50\
    --save_every 350\
    --learning_rate 2e-4\
    --margin 0.9\
    --save_path_prefix /content/drive/MyDrive/outputs/want_path
  

