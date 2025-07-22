CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python /content/drive/MyDrive/UnLoG/UnLoG_training/pretraining/train.py\
    --model_name bert-large-uncased\
    --train_path /content/drive/MyDrive/UnLoG/UnLoG_training/data/redone_3col_oddX.csv\
    --dev_path /content/drive/MyDrive/UnLoG/UnLoG_training/data/wikitext103_raw_v1_validation.txt\
    --seqlen 64\
    --number_of_gpu 1\
    --batch_size_per_gpu 16\
    --gradient_accumulation_steps 2\
    --effective_batch_size 32\
    --total_steps 630\
    --print_every 50\
    --save_every 630\
    --learning_rate 4e-5\
    --margin 0.9\
    --save_path_prefix /content/drive/MyDrive/outputs/want_path
  

