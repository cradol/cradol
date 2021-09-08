import os
import time
import torch
import random
import numpy as np
from gym_env import make_envs
from logger import TensorBoardLogger
from misc.utils import set_log, load_config
from misc.arguments import args
from trainer import train
from algorithms.soc.agent import SoftOptionCritic
from algorithms.soc.replay_buffer import ReplayBufferSOC

torch.set_num_threads(3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    # Create directories
    if not os.path.exists("./logs/"):
        os.makedirs("./logs/")
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Create loggings
    log = set_log(args)
    tb_writer = TensorBoardLogger(logdir="./logs_tensorboard/", run_name=args.log_name + time.ctime())

    # Make env
    env = make_envs(args.env_name, args)

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    if device == torch.device("cuda"):
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(args.seed)

    # Set SOC
    agent = SoftOptionCritic(env=env, args=args, tb_writer=tb_writer, log=log)
    replay_buffer = ReplayBufferSOC(
        env=env, size=args.buffer_size, hidden_size=args.hidden_size,
        rim_num=args.rim_num, value_size=args.value_size)

    if args.load_model:
        model_dict = torch.load(args.model_name)
        agent.load_state_dict(model_dict)

    # Set training
    train(agent=agent, env=env, replay_buffer=replay_buffer, args=args)


if __name__ == '__main__':
    # Load experiment specific config if provided
    if args.config is not None:
        load_config(args)

    # Set log name
    args.log_name = \
        "%s_env::%s_seed::%s_lr::%s_alpha::%s_option_num" \
        "::%s_num_units::%s_k::%s_hidden_size::%s_batch_size::%s_value_size" % (
            args.exp_name, args.seed, args.lr, args.alpha, args.option_num,
            args.num_units, args.k, args.hidden_size, args.batch_size, args.value_size)

    main(args)
