#!/usr/bin/env bash
set -x
PY_ARGS=${@}


python main.py \
    --target_dataset mnist_m \
    --lr 0.001 \
    --batch_size 128 \
    --epochs 50 \
    --seed 42 \
    --amp ${PY_ARGS}