#!/bin/bash
n_gpu=4
cls='ape'
python -m torch.distributed.launch --nproc_per_node=4 train_lm.py --gpu "0, 1, 2, 3" --gpus $n_gpu --cls=$cls --opt_level "O1" --gpu_id [0,1,2,3]
