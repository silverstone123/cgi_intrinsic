#!/bin/bash

PYTHON_PATH=./my_python
SAVE_PATH=/net/acadia6a/data/xiangyu/pytorch_model/intrinsic

source $PYTHON_PATH/bin/activate

nohup srun --constraint=GPU12GB \
  --mem=48G \
  --gres=gpu:1 \
  --time=10-5 \
  --job-name=lessD_64 \
  --output=./log/10_17_2018_coarse_64.log \
  --exclude=skyserver5k,skyserver6k \
  $PYTHON_PATH/bin/python trainCGI_coarse_lessData.py \
  $SAVE_PATH/10_17_2018_coarse_64 \
  1e-3 \
  0.00001 \
  100 &

