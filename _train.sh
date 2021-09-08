#!/bin/bash

# Load python virtualenv
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt

# For MuJoCo domains
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$USER/.mujoco/mujoco200/bin
echo $LD_LIBRARY_PATH

# Comment if using GPU
export CUDA_VISIBLE_DEVICES=-1

# Begin experiment
python3 main.py --config moving_bandit.yaml
