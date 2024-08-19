#!/usr/bin/env bash

eval "$(conda shell.bash hook)"
conda activate exp_pl

CUDA_VISIBLE_DEVICES=1 python -W ignore src/train.py \
  experiment=hmdb51 \
  data.batch_size=1 \
  trainer=gpu \
  trainer.devices=1

conda deactivate