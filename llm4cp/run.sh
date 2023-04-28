#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=True

function train_actor() {
    deepspeed actor.py --deepspeed
}

function train_ppo() {
    deepspeed ppo.py \
        --deepspeed \
        --use_real_reward true \
        --use_offload true
}

train_actor
#train_ppo
