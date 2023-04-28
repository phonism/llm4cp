#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=True


deepspeed actor.py \
    --deepspeed 
    #--use_offload true
