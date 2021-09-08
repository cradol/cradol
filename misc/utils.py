import torch
import logging
import yaml
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.special import softmax


def load_config(args, path="."):
    """Loads and replaces default parameters with experiment
    specific parameters

    Args:
        args (argparse): Python argparse that contains arguments
        path (str): Root directory to load config from. Default: "."
    """
    with open(path + "/config/" + args.config, 'r') as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        args.__dict__[key] = value


def set_logger(logger_name, log_file, level=logging.INFO):
    log = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    log.setLevel(level)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)
    log.propagate = False  # otherwise root logger prints things again


def set_log(args):
    log = {}
    set_logger(
        logger_name=args.log_name,
        log_file=r'{0}{1}'.format("./logs/", args.log_name))
    log[args.log_name] = logging.getLogger(args.log_name)

    for arg, value in sorted(vars(args).items()):
        log[args.log_name].info("%s: %r", arg, value)

    return log


def create_array(env, grid_size):
    original_env = str(env)
    count, i = 0, 0
    arr = np.zeros((grid_size, grid_size))
    while(i < len(original_env)):
        r = count // grid_size
        c = count % grid_size
        value = original_env[i]
        if value == "W" or value == "G":
            arr[r][c] = 0.0
            i += 2
            count += 1
        elif value == "L":
            arr[r][c] = 0.0
            i += 2
            count += 1
        elif value == "K":
            arr[r][c] = 0.0
            i += 2
            count += 1
        elif value == "<" or value == ">" or value == "V" or value == "^":
            arr[r][c] = 0.0
            i += 2
            count += 1
        elif value == "\n" or value == "\n ":
            i += 1
        else:
            arr[r][c] = 0.0
            i += 2
            count += 1

    return arr


def update_array(env, arr, grid_size, option):
    original_env = str(env)
    count, i = 0, 0
    option_lookup = {0: 0.25, 1: 0.5, 2: 0.75}
    while(i < len(original_env)):
        r = count // grid_size
        c = count % grid_size
        value = original_env[i]
        if value == "<" or value == ">" or value == "V" or value == "^":
            arr[r][c] = option_lookup[option.item()]
            i += 2
            count += 1
            break
        elif value == "\n" or value == "\n ":
            i += 1
        else:
            i += 2
            count += 1
    return arr


def visualize(arr, count, name):
    if name == "grid":
        plt.imshow(arr, cmap="gray")
        plt.savefig(name + "_" + str(count) + ".svg")


def visualize_both(arr, render_env, count):
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(arr, cmap="gray")
    axarr[1].imshow(render_env)
    plt.savefig("grid" + "_" + str(count) + ".svg")


def visualize_lookup(lookup, count):
    font = {'family': 'Calibri',
            'weight': 'bold',
            'size': 12}

    matplotlib.rc('font', **font)
    lookup = softmax(lookup, axis=1)
    plt.clf()
    sns_plot = sns.heatmap(lookup, annot=True, cmap="Greys", xticklabels=False, yticklabels=False)

    fig_seaborn = sns_plot.get_figure()

    fig_seaborn.savefig("option_prob" + "_" + str(count) + ".svg")
    fig_seaborn.clf()


def get_factored_corr(agent, option_num, obs, state_in_pi):
    factored_states = []
    for i in range(option_num):
        hidden = agent.visualization_output(obs, i, state_in_pi).detach().numpy()
        factored_states.append(hidden)

    corr = np.corrcoef(factored_states)
    return corr


def visualize_factor(corr, count):
    plt.clf()
    corr_option = np.mean(corr, axis=0)
    sns_plot = sns.heatmap(corr_option, annot=True, cmap="crest", xticklabels=False, yticklabels=False)

    fig_seaborn = sns_plot.get_figure()
    fig_seaborn.savefig("visual_factor" + str(count) + ".svg")
    fig_seaborn.clf()


def process_done_signal(env, ep_len, reward, done, args):
    done = False if ep_len == env.max_episode_steps else done
    if args.env_name == "Maze" and reward == 1:
        done = True

    return done


def initialize_lstm_state(args):
    hidden_in_pi = torch.zeros(args.rim_num, args.hidden_size)
    cell_in_pi = torch.zeros(args.rim_num, args.hidden_size)

    hidden_in_beta = torch.zeros(args.hidden_size)
    cell_in_beta = torch.zeros(args.hidden_size)

    hidden_in_inter_q1 = torch.zeros(args.hidden_size)
    cell_in_inter_q1 = torch.zeros(args.hidden_size)

    hidden_in_inter_q2 = torch.zeros(args.hidden_size)
    cell_in_inter_q2 = torch.zeros(args.hidden_size)

    hidden_in_intra_q1 = torch.zeros(args.hidden_size)
    cell_in_intra_q1 = torch.zeros(args.hidden_size)

    hidden_in_intra_q2 = torch.zeros(args.hidden_size)
    cell_in_intra_q2 = torch.zeros(args.hidden_size)

    return \
        (hidden_in_pi, cell_in_pi), \
        (hidden_in_beta, cell_in_beta), \
        (hidden_in_inter_q1, cell_in_inter_q1), \
        (hidden_in_inter_q2, cell_in_inter_q2), \
        (hidden_in_intra_q1, cell_in_intra_q1), \
        (hidden_in_intra_q2, cell_in_intra_q2)
