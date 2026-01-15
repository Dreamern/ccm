#!/bin/bash
torchrun --nproc_per_node=$1 --master_port=29505 train_ccm.py \
    --model otcfm \
    --output_dir results/ccm \
    --lr 2e-4 \
    --total_steps 300001 \
    --warmup 5000 \
    --batch_size 128 \
    --save_step 20000 \
    --resume null \
    --dataset cifar10 \
    --kdc 65 \
    --gan_start 100000 \
    --alpha 2 \
    --loss_type 224 \
    --cm_type cd \
    --timestep_size 0.03 \
    --teacher results/fm/otcfm/cifar10/otcfm_cifar10_weights_step_400000.pt