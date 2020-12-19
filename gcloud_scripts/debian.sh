#!/usr/bin/env bash

# Simple test for the debian-based boxes on Google cloud, will remove this file in future version
# For gcloud_scripts to provision and deprovision as well as install and test, see:
# https://github.com/SamuelMarks/ml-glaucoma/tree/master/gcloud_scripts

set -veuo pipefail

sudo apt-get update
sudo apt-get install -y python3-venv
python3 -m venv venv
. ~/venv/bin/activate
pip3 install -U pip setuptools wheel
pip3 install tensorflow tensorflow-datasets jupyter
pip install -r 'https://raw.githubusercontent.com/SamuelMarks/ml-params-tensorflow/master/requirements.txt'
pip install 'https://api.github.com/repos/SamuelMarks/ml-params-tensorflow/zipball#egg=ml-params-tensorflow'

if [ -z "$TPU_ADDR" ]; then
    printf '$TPU_ADDR must be set\n' >&2
    exit 1
else
  printf 'Training on: %s\n' "$TPU_ADDR"
fi

export ML_PARAMS_ENGINE='tensorflow'
python -m ml_params load_data --dataset_name 'cifar10' \
                    load_model --model 'MobileNet' \
                    train --epochs 3 \
                    --loss 'BinaryCrossentropy' \
                    --optimizer 'Adam' \
                    --callbacks 'TensorBoard: --log_dir "/tmp"' \
                    --metrics 'binary_accuracy' \
                    --tpu_address "$TPU_ADDR"
