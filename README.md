# CRADOL
Source code for "Context-Specific Representation Abstraction for Deep Option Learning"

## Dependency
Known dependencies are (please also refer to `requirements.txt`):
```
python 3.6.5
pip3.6
numpy>=1.14.0
torch==1.9.0+cu111
gym==0.19.0
tensorboardX==1.2
pyyaml==5.4.1
matplotlib
seaborn==0.11.1
gym_minigrid
mujoco-py<2.1,>=2.0
```

## Setup
To avoid any conflict, please install virtual environment with [`virtualenv`](http://docs.python-guide.org/en/latest/dev/virtualenvs/):
```
pip3.6 install --upgrade virtualenv
```
Please note that all the required dependencies will be automatically installed in the virtual environment by running the training script (`_train.sh`).

## Run
To start training in moving bandit domain (default):
```
./_train.sh
```

Additionally, to see the tensorboard logging during training:
```
tensorboard --logdir=logs_tensorboard
```

## Credit
This repository is built on based on the following helpful implementations:
* [RIMs](https://github.com/dido1998/Recurrent-Independent-Mechanisms)
* [Moving bandit](https://github.com/maximilianigl/rl-msol)
* [Reacher](https://github.com/openai/gym/blob/master/gym/envs/mujoco/reacher.py)
