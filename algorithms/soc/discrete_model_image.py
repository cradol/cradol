import torch
import torch.multiprocessing
import torch.nn.functional as F
from torch import nn as nn
from algorithms.soc.RIM import RIMCell
from misc.torch_utils import initialize


class InterQFunction(torch.nn.Module):
    def __init__(self, obs_dim, hidden_size, option_dim):
        super(InterQFunction, self).__init__()

        self.inter_q_pre_fc = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_dim[0]
        m = obs_dim[1]

        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64
        self.inter_q_rnn = nn.LSTM(self.image_embedding_size, hidden_size)
        self.inter_q_layer = nn.Linear(hidden_size, option_dim)

    def forward(self, input):
        x, (hx, cx) = input
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        x = x.transpose(1, 3).transpose(2, 3)

        x = self.inter_q_pre_fc(x)
        x = x.reshape(x.shape[0], -1)

        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        if len(hx.shape) == 1:
            hx = hx.unsqueeze(0).unsqueeze(0)
            cx = cx.unsqueeze(0).unsqueeze(0)
        else:
            hx = hx.unsqueeze(0)
            cx = cx.unsqueeze(0)

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
    def __init__(self, obs_dim, hidden_size, option_dim, action_dim):
        super(IntraQFunction, self).__init__()

        self.intra_q_pre_fc = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_dim[0]
        m = obs_dim[1]

        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64
        self.intra_q_rnn = nn.LSTM(self.image_embedding_size, hidden_size)
        self.intra_q_layer = nn.Linear(hidden_size + option_dim, action_dim)

    def forward(self, input, option):
        x, (hx, cx) = input
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        x = x.transpose(1, 3).transpose(2, 3)

        x = self.intra_q_pre_fc(x)
        x = x.reshape(x.shape[0], -1)

        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        if len(hx.shape) == 1:
            hx = hx.unsqueeze(0).unsqueeze(0)
            cx = cx.unsqueeze(0).unsqueeze(0)
        else:
            hx = hx.unsqueeze(0)
            cx = cx.unsqueeze(0)

        _, (hy, cy) = self.intra_q_rnn(x, (hx, cx))

        if len(hy.shape) == 3:
            hy = hy.squeeze(0).squeeze(0)
            cy = cy.squeeze(0).squeeze(0)

        if len(hy.shape) == 2:
            hy = hy.squeeze(0)
            cy = cy.squeeze(0)

        x = hy

        x = torch.cat([x, option], dim=-1)

        q = self.intra_q_layer(x).squeeze(-1)
        return q, (hy, cy)


class IntraOptionPolicy(torch.nn.Module):
    def __init__(self, obs_dim, hidden_size, option_dim, action_dim, rim_units, k, value_size):
        super(IntraOptionPolicy, self).__init__()

        self.pi_pre_fc = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_dim[0]
        m = obs_dim[1]
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64
        self.rim_units = rim_units
        self.k = k

        self.pi_rnn = RIMCell(
            torch.device('cpu'), self.image_embedding_size, hidden_size,
            self.rim_units, self.k, 'LSTM', option_num=option_dim, input_value_size=value_size,
            comm_value_size=self.image_embedding_size // self.rim_units)

        self.pi_w_layer = torch.randn((option_dim, hidden_size, action_dim))
        self.pi_b_layer = torch.randn((option_dim, action_dim))

        self.pi_w_layer, self.pi_b_layer = initialize(self.pi_w_layer, self.pi_b_layer)

    def forward(self, input, option):
        x, (hx, cx) = input

        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if len(option.shape) == 1:
            option = option.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif len(option.shape) == 2:
            option = option.unsqueeze(0).unsqueeze(0)

        x = x.transpose(1, 3).transpose(2, 3)
        x = self.pi_pre_fc(x)
        x = x.reshape(x.shape[0], -1)

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        if len(hx.shape) == 2:
            hx = hx.unsqueeze(0)
            cx = cx.unsqueeze(0)

        hy, cy = self.pi_rnn(x=x, hs=hx, cs=cx, option=option)

        x = hy

        x = torch.einsum("buh, oha -> boa", x, self.pi_w_layer)
        x = torch.add(x, self.pi_b_layer)
        action_probs = F.softmax(x, dim=-1)

        return action_probs, (hy.squeeze(0), cy.squeeze(0))


class BetaPolicy(torch.nn.Module):
    def __init__(self, value_size, hidden_size, option_dim):
        super(BetaPolicy, self).__init__()

        self.beta_pre_fc = nn.Sequential(nn.Linear(value_size + option_dim, hidden_size), nn.ReLU())
        self.beta_rnn = nn.LSTM(hidden_size, hidden_size)
        self.beta_layer = nn.Linear(hidden_size, 1)

    def forward(self, input, option):
        x, (hx, cx) = input
        x = torch.cat([x, option], dim=-1)

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


class SOCModelImageCategorical(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size, option_dim, rim_units, k, value_size):
        super(SOCModelImageCategorical, self).__init__()

        # Inter-Q Function Definitions
        self.inter_q_function_1 = InterQFunction(obs_dim, hidden_size, option_dim)
        self.inter_q_function_2 = InterQFunction(obs_dim, hidden_size, option_dim)

        # Intra-Q Function Definitions
        self.intra_q_function_1 = IntraQFunction(obs_dim, hidden_size, option_dim, action_dim)
        self.intra_q_function_2 = IntraQFunction(obs_dim, hidden_size, option_dim, action_dim)

        # Policy Definitions
        self.intra_option_policy = IntraOptionPolicy(obs_dim, hidden_size, option_dim, action_dim, rim_units, k, value_size)

        # Beta Definitions
        self.beta_policy = BetaPolicy(value_size, hidden_size, option_dim)
