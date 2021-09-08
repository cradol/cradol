import torch
import torch.multiprocessing
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch import nn as nn

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class InterQFunction(torch.nn.Module):
    def __init__(self, obs_dim, hidden_size, option_dim):
        super(InterQFunction, self).__init__()

        self.inter_q_pre_fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU())

        self.inter_q_rnn = nn.LSTM(hidden_size, hidden_size)
        self.inter_q_layer = nn.Linear(hidden_size, option_dim)

    def forward(self, input):
        x, (hx, cx) = input

        if len(x.shape) == 1:
            x = x.unsqueeze(0).unsqueeze(0)
            hx = hx.unsqueeze(0).unsqueeze(0)
            cx = cx.unsqueeze(0).unsqueeze(0)

        elif len(x.shape) == 2:
            x = x.unsqueeze(0)
            hx = hx.unsqueeze(0)
            cx = cx.unsqueeze(0)

        assert len(x.shape) == 3, "Shape must be 3 for RNN input"
        x = self.inter_q_pre_fc(x)
        _, (hy, cy) = self.inter_q_rnn(x, (hx, cx))

        if len(hy.shape) == 3:
            hy = hy.squeeze(0).squeeze(0)
            cy = cy.squeeze(0).squeeze(0)

        if len(hy.shape) == 2:
            hy = hy.squeeze(0)
            cy = cy.squeeze(0)

        x = hy
        q = self.inter_q_layer(x)
        return q, (hy, cy)


class IntraQFunction(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, option_dim, hidden_size):
        super(IntraQFunction, self).__init__()

        self.intra_q_pre_fc = nn.Sequential(
            nn.Linear(obs_dim + option_dim + act_dim, hidden_size),
            nn.ReLU())

        self.intra_q_rnn = nn.LSTM(hidden_size, hidden_size)
        self.intra_q_layer = nn.Linear(hidden_size, 1)

    def forward(self, input, option, action):
        x, (hx, cx) = input
        x = torch.cat([x, option, action], dim=-1)

        if len(x.shape) == 1:
            x = x.unsqueeze(0).unsqueeze(0)
            hx = hx.unsqueeze(0).unsqueeze(0)
            cx = cx.unsqueeze(0).unsqueeze(0)

        elif len(x.shape) == 2:
            x = x.unsqueeze(0)
            hx = hx.unsqueeze(0)
            cx = cx.unsqueeze(0)

        assert len(x.shape) == 3, "Shape must be 3 for RNN input"
        x = self.intra_q_pre_fc(x)

        _, (hy, cy) = self.intra_q_rnn(x, (hx, cx))

        if len(hy.shape) == 3:
            hy = hy.squeeze(0).squeeze(0)
            cy = cy.squeeze(0).squeeze(0)

        if len(hy.shape) == 2:
            hy = hy.squeeze(0)
            cy = cy.squeeze(0)

        x = hy
        q = self.intra_q_layer(x).squeeze(-1)
        return q, (hy, cy)


class IntraOptionPolicy(torch.nn.Module):
    """
    There will be n of these policies where n is the number of options
    Input: state
    Output: number of actions
    """

    def __init__(self, obs_dim, act_dim, option_dim, hidden_size, act_limit):
        super(IntraOptionPolicy, self).__init__()
        self.pi_pre_fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU())

        self.pi_rnn = nn.LSTM(hidden_size, hidden_size)

        self.w_mu_layer = torch.randn((option_dim, hidden_size, act_dim))
        self.w_log_std_layer = torch.randn((option_dim, hidden_size, act_dim))

        self.b_mu_layer = torch.randn((option_dim, act_dim))
        self.b_log_std_layer = torch.randn((option_dim, act_dim))
        self.act_limit = act_limit
        self.option_dim = option_dim
        self.act_dim = act_dim

    def forward(self, input):
        x, (hx, cx) = input

        if len(x.shape) == 1:
            x = x.unsqueeze(0).unsqueeze(0)
            hx = hx.unsqueeze(0).unsqueeze(0)
            cx = cx.unsqueeze(0).unsqueeze(0)

        elif len(x.shape) == 2:
            x = x.unsqueeze(0)
            hx = hx.unsqueeze(0)
            cx = cx.unsqueeze(0)

        x = self.pi_pre_fc(x)
        assert len(x.shape) == 3, "Shape must be 3 for RNN input"

        _, (hy, cy) = self.pi_rnn(x, (hx, cx))

        if len(hy.shape) == 3:
            hy = hy.squeeze(0).squeeze(0)
            cy = cy.squeeze(0).squeeze(0)

        if len(hy.shape) == 2:
            hy = hy.squeeze(0)
            cy = cy.squeeze(0)

        x = hy
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        mu = torch.einsum("bh, oha -> boa", x, self.w_mu_layer)
        log_std = torch.einsum("bh, oha -> boa", x, self.w_log_std_layer)
        assert mu.shape == (x.shape[0], self.option_dim, self.act_dim)

        mu = torch.add(mu, self.b_mu_layer)
        log_std = torch.add(log_std, self.b_log_std_layer)

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        pi_distribution = Normal(loc=mu, scale=std)
        pi_action = pi_distribution.rsample()
        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)

        logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=-1)
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        return pi_action, logp_pi, (hy, cy)


class BetaPolicy(torch.nn.Module):
    def __init__(self, obs_dim, hidden_size, option_dim):
        super(BetaPolicy, self).__init__()

        self.beta_pre_fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU())

        self.beta_rnn = nn.LSTM(hidden_size, hidden_size)

        self.beta_layer = nn.Linear(hidden_size, option_dim)

    def forward(self, input):
        x, (hx, cx) = input

        if len(x.shape) == 1:
            x = x.unsqueeze(0).unsqueeze(0)
            hx = hx.unsqueeze(0).unsqueeze(0)
            cx = cx.unsqueeze(0).unsqueeze(0)

        elif len(x.shape) == 2:
            x = x.unsqueeze(0)
            hx = hx.unsqueeze(0)
            cx = cx.unsqueeze(0)

        assert len(x.shape) == 3, "Shape must be 3 for RNN input"
        x = self.beta_pre_fc(x)

        _, (hy, cy) = self.beta_rnn(x, (hx, cx))

        if len(hy.shape) == 3:
            hy = hy.squeeze(0).squeeze(0)
            cy = cy.squeeze(0).squeeze(0)

        if len(hy.shape) == 2:
            hy = hy.squeeze(0)
            cy = cy.squeeze(0)

        x = hy
        beta = torch.sigmoid(self.beta_layer(x))
        return beta, (hy, cy)


class SOCModelContinous(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size, option_dim, act_limit):
        super(SOCModelContinous, self).__init__()
        # Inter-Q Function Definitions
        self.inter_q_function_1 = InterQFunction(obs_dim, hidden_size, option_dim)
        self.inter_q_function_2 = InterQFunction(obs_dim, hidden_size, option_dim)

        # Intra-Q Function Definitions
        self.intra_q_function_1 = IntraQFunction(obs_dim, act_dim, option_dim, hidden_size)
        self.intra_q_function_2 = IntraQFunction(obs_dim, act_dim, option_dim, hidden_size)

        # Policy Definitions
        self.intra_option_policy = IntraOptionPolicy(obs_dim, act_dim, option_dim, hidden_size, act_limit)
