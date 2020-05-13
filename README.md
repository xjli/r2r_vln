# Robust Navigation with Language Pretraining and Stochastic Sampling 

## Introduction
This repository contains source code and trained checkpoint to reproduce the results presented in the paper [Robust Navigation with Language Pretraining and Stochastic Sampling](https://arxiv.org/abs/1909.02244).


## Download
We provide two pre-trained models of Bert-base.
```bash
wget https://xiuldlstorage.blob.core.windows.net/r2r/public/emnlp/$MODEL_NAME.zip
unzip $MODEL_NAME.zip -d $MODEL_DIR
```
`MODEL_NAME` could be `spl_53`, `spl_54.4`.


## Installation
Please follow [R2R](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/tasks/R2R) for the environment setup.

## Run Training
Script to run training.
```bash
TBA
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