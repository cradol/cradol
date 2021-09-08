import torch
import torch.multiprocessing
import numpy as np
import torch.nn as nn

# Setting CUDA USE
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda" if use_cuda else "cpu")


def to_onehot(value, dim):
    """Convert batch of numbers to onehot

    Args:
        value (numpy.ndarray): Batch of numbers to convert to onehot. Shape: (batch,)
        dim (int): Dimension of onehot
    Returns:
        onehot (numpy.ndarray): Converted onehot. Shape: (batch, dim)
    """
    value = value.squeeze(-1)
    one_hot = torch.zeros(value.shape[0], dim)
    one_hot[torch.arange(value.shape[0]), value.long()] = 1
    return one_hot


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x ** 2).sum(1, keepdim=True))
    return x


def ensure_shared_grads(model, shared_model, gpu=False):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None and not gpu:
            return
        elif not gpu:
            shared_param._grad = param.grad
        else:
            shared_param._grad = param.grad.cpu()


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


def tensor(x):
    if isinstance(x, torch.Tensor):
        return x.to(device).float()
    x = torch.tensor(x, device=device)
    return x.to(device).float()


def convert_onehot(x, dim):
    x = x.cpu().flatten().numpy().astype(int)
    one_hot = np.zeros((x.size, dim))
    rows = np.arange(len(x))
    one_hot[rows, x] = 1
    return one_hot


def initialize(weight, bias):
    """Initialize layer weight based on Xavier normal
    Only supported layer types are nn.Linear and nn.LSTMCell
    Args:
        module (class): Layer to initialize weight, including bias
    """

    weight = nn.init.xavier_normal_(weight)
    bias = bias.data.zero_()
    return weight, bias


def softmax(x):
    """ applies softmax to an input x"""
    e_x = np.exp(x)
    return e_x / e_x.sum()
