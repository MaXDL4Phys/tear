#!/usr/bin/env bash

eval "$(conda shell.bash hook)"
conda activate myenv

for t in 0.2 0.6 1.2 1.6 2.2 2.6
do
  CUDA_VISIBLE_DEVICES=0 python -W ignore ../src/eval.py \
    experiment=$s \
    model.method=$m \
    model.decomposition.input_conditioning=True \
    model.decomposition.alpha=0.8 \
    model.network.temperature=$t \
    data.batch_size=4 \
    logger.wandb.offline=True \
    logger.wandb.tags=["t_"$t""] \
    logger.wandb.project=debug \
    trainer=gpu \
    trainer.devices=1
done

conda deactivate