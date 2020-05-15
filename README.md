# Robust Navigation with Language Pretraining and Stochastic Sampling 

## Introduction
This repository contains source code and trained checkpoint to reproduce the results presented in the paper [Robust Navigation with Language Pretraining and Stochastic Sampling](https://arxiv.org/abs/1909.02244).


## Download
We provide two trained model checkpoints of Bert-base.
```bash
wget https://xiuldlstorage.blob.core.windows.net/r2r/public/emnlp/$MODEL_NAME.zip
unzip $MODEL_NAME.zip -d $MODEL_DIR
```
`MODEL_NAME` could be `spl_53`, `spl_54.4`.

Download the generated paths for data augmentation.
```bash
wget https://xiuldlstorage.blob.core.windows.net/r2r/public/emnlp/data/R2R_bi_12700_seed10-60_literal_speaker_data_aug_paths_unk_bert.txt
```

Download the bi-directional speaker generated paths for data augmentation.
```bash
wget https://xiuldlstorage.blob.core.windows.net/r2r/public/emnlp/data/R2R_bi_12700_seed10-60_literal_speaker_data_augmentation_paths_bert.txt
```

Download some pre-trained Bert base and large checkpoints for play.
```bash
wget https://xiuldlstorage.blob.core.windows.net/r2r/public/emnlp/pretrain.zip
```

## Installation
Please follow [R2R](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/tasks/R2R) for the environment setup.
```bash
# install bert package
pip install pytorch-pretrained-bert
```

## Run Training
Script to run training (not update Bert, transformer_update=False).
```bash
# For example, best val_unseen spl = 0.53, best val_unseen sr = 0.58
CUDA_VISIBLE_DEVICES=0 python ./tasks/R2R/train.py --feedback_method teacher --bidirectional True --encoder_type bert --top_lstm True --transformer_update False --batch_size 20 --log_every 40 --pretrain_n_sentences 6 --pretrain_splits bi_12700_seed10-60_literal_speaker_data_aug_paths_unk --save_ckpt 10000 --ss_n_pretrain_iters 50000 --pretrain_n_iters 60000 --ss_n_iters 60000 --n_iters 70000 --dropout_ratio 0.4 --dec_h_type vc --schedule_ratio 0.4 --optm Adamax --att_ctx_merge mean --clip_gradient_norm 0 --clip_gradient 0.1 --use_pretrain --action_space -1 --pretrain_score_name sr_unseen --train_score_name sr_unseen --enc_hidden_size 1024 --hidden_size 1024 --result_dir ./base/results/ --snapshot_dir ./base/snapshots/ --plot_dir ./base/plots/
```

Script to run finetuning (update Bert, transformer_update=True).
```bash
CUDA_VISIBLE_DEVICES=0 python ./tasks/R2R/train.py --feedback_method teacher --dropout_ratio 0.4 --dec_h_type vc --optm Adamax --schedule_ratio 0.2 --att_ctx_merge mean --clip_gradient_norm 0 --clip_gradient 0.1 --log_every 32 --action_space -1 --n_iters 34000 --train_score_name sr_unseen --enc_hidden_size 1024 --hidden_size 1024 --result_dir ./base/results/ --snapshot_dir ./base/snapshots/ --plot_dir ./base/plots/ --n_iters_resume N --ss_n_iters N+10000 --save_ckpt 512 --bidirectional True --encoder_type bert --top_lstm True --transformer_update True --batch_size 16 --learning_rate 5e-5

N is the iteration of the best checkpoint from the above training.

# For example, with a pre-train base checkpoint, N = 61040, best val_unseen spl = 0.556, best val_unseen sr = 0.602
CUDA_VISIBLE_DEVICES=0 python ./tasks/R2R/train.py --feedback_method teacher --n_iters_pretrain_resume 61040 --learning_rate 0.0001 --pretrain_model_path path_to/pretrain/bert_s/ --save_ckpt 9600 --ss_n_pretrain_iters 71000 --pretrain_n_iters 81000 --ss_n_iters 81000 --n_iters 91000 --bidirectional True --encoder_type bert --top_lstm True --bert_n_layers 1 --transformer_update True --batch_size 16 --log_every 48 --pretrain_n_sentences 6 --pretrain_splits bi_12700_seed10-60_literal_speaker_data_augmentation_paths --dropout_ratio 0.4 --dec_h_type vc --schedule_ratio 0.3 --optm Adamax --att_ctx_merge mean --clip_gradient_norm 0 --clip_gradient 0.1 --use_pretrain --action_space -1 --pretrain_score_name sr_unseen --train_score_name spl_unseen --enc_hidden_size 1024 --hidden_size 1024
```


## Run Validation and Test
Script to play with the checkpoint on the paper (val unseen spl=55).
```bash
CUDA_VISIBLE_DEVICES=0 python ./tasks/R2R/train.py --panoramic True --result_dir ./test --snapshot_dir ./snapshots --plot_dir ./plot --action_space -1 --n_iters 10 --att_ctx_merge mean --n_iters_resume 63480 --sc_after 0 --sc_score_name sr_unseen --train False --val_splits val_seen,val_unseen,test --enc_hidden_size 1024 --hidden_size 1024 --feedback_method teacher --clip_gradient 0.1 --clip_gradient_norm 0 --dec_h_type vc --schedule_ratio -1.0 --dump_result --bidirectional True --optm Adamax --encoder_type bert --top_lstm True --transformer_update False --batch_size 24 --pretrain_model_path path_to/spl_53/snapshots/
```

Script to play with a better checkpoint (val unseen spl=56.2).
```bash
CUDA_VISIBLE_DEVICES=0 python ./tasks/R2R/train.py --panoramic True --result_dir ./test --snapshot_dir ./snapshots --plot_dir ./plot --action_space -1 --n_iters 10 --att_ctx_merge mean --n_iters_resume 68576 --sc_after 0 --sc_score_name sr_unseen --train False --val_splits val_seen,val_unseen,test --enc_hidden_size 1024 --hidden_size 1024 --feedback_method teacher --clip_gradient 0.1 --clip_gradient_norm 0 --dec_h_type vc --schedule_ratio -1.0 --dump_result --bidirectional True --optm Adamax --encoder_type bert --top_lstm True --transformer_update False --batch_size 24 --pretrain_model_path path_to/spl_54.4/snapshots/
```


## Citations
Please consider citing this paper if you use the code:
```
@article{li2019robust,
  title={Robust Navigation with Language Pretraining and Stochastic Sampling},
  author={Li, Xiujun and Li, Chunyuan and Xia, Qiaolin and Bisk, Yonatan and Celikyilmaz, Asli and Gao, Jianfeng and Smith, Noah and Choi, Yejin},
  conference={EMNLP},
  year={2019}
}
```