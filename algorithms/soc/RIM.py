import math
import torch
import torch.nn as nn
import numpy as np
from misc.torch_utils import convert_onehot, initialize


class blocked_grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(x, mask)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_tensors
        return grad_output * mask, mask * 0.0


class GroupLinearLayer(nn.Module):
    def __init__(self, din, dout, num_blocks):
        super(GroupLinearLayer, self).__init__()

        self.w = nn.Parameter(0.01 * torch.randn(num_blocks, din, dout))

    def forward(self, x):
        x = x.permute(1, 0, 2)

        x = torch.bmm(x, self.w)
        return x.permute(1, 0, 2)


class GroupLSTMCell(nn.Module):
    """
    GroupLSTMCell can compute the operation of N LSTM Cells at once.
    """
    def __init__(self, inp_size, hidden_size, num_lstms):
        super().__init__()
        self.inp_size = inp_size
        self.hidden_size = hidden_size

        self.i2h = GroupLinearLayer(inp_size, 4 * hidden_size, num_lstms)
        self.h2h = GroupLinearLayer(hidden_size, 4 * hidden_size, num_lstms)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hid_state):
        """
        input: x (batch_size, num_lstms, input_size)
               hid_state (tuple of length 2 with each element of size (batch_size, num_lstms, hidden_state))
        output: h (batch_size, num_lstms, hidden_state)
                c ((batch_size, num_lstms, hidden_state))
        """
        h, c = hid_state
        preact = self.i2h(x) + self.h2h(h)

        gates = preact[:, :, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, :, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :, :self.hidden_size]
        f_t = gates[:, :, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, :, -self.hidden_size:]

        c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t)
        h_t = torch.mul(o_t, c_t.tanh())

        return h_t, c_t


class GroupGRUCell(nn.Module):
    """
    GroupGRUCell can compute the operation of N GRU Cells at once.
    """
    def __init__(self, input_size, hidden_size, num_grus):
        super(GroupGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x2h = GroupLinearLayer(input_size, 3 * hidden_size, num_grus)
        self.h2h = GroupLinearLayer(hidden_size, 3 * hidden_size, num_grus)
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.parameters():
            w.data = torch.ones(w.data.size())  # .uniform_(-std, std)

    def forward(self, x, hidden):
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        i_r, i_i, i_n = gate_x.chunk(3, 2)
        h_r, h_i, h_n = gate_h.chunk(3, 2)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)

        return hy


class RIMCell(nn.Module):
    def __init__(self, device, input_size, hidden_size, num_units, k, rnn_cell, option_num, input_key_size=64,
                 input_value_size=400, input_query_size=64, num_input_heads=1, input_dropout=0.1, comm_key_size=32,
                 comm_value_size=100, comm_query_size=32, num_comm_heads=4, comm_dropout=0.1):
        super().__init__()

        if comm_value_size != hidden_size:
            comm_value_size = hidden_size

        self.input_size = input_size
        self.device = device
        self.hidden_size = hidden_size
        self.num_units = num_units
        self.rnn_cell = rnn_cell
        self.key_size = input_key_size
        self.k = k
        self.num_input_heads = num_input_heads
        self.num_comm_heads = num_comm_heads
        self.input_key_size = input_key_size
        self.input_query_size = input_query_size
        self.input_value_size = input_value_size

        self.comm_key_size = comm_key_size
        self.comm_query_size = comm_query_size
        self.comm_value_size = comm_value_size
        self.option_num = option_num

        # Input Attention
        self.value = nn.Linear(self.input_size, self.num_input_heads * self.input_value_size)

        self.p_w = torch.randn(self.option_num, self.num_units)
        self.p_b = torch.randn(self.num_units)
        self.p_w, self.p_b = initialize(self.p_w, self.p_b)

        if self.rnn_cell == 'GRU':
            self.rnn = GroupGRUCell(input_value_size, hidden_size, num_units)
        else:
            self.rnn = GroupLSTMCell(input_value_size, hidden_size, num_units)

        self.query_ = GroupLinearLayer(hidden_size, comm_query_size * num_comm_heads, self.num_units)
        self.key_ = GroupLinearLayer(hidden_size, comm_key_size * num_comm_heads, self.num_units)
        self.value_ = GroupLinearLayer(hidden_size, comm_value_size * num_comm_heads, self.num_units)

        self.comm_attention_output = GroupLinearLayer(num_comm_heads * comm_value_size, comm_value_size, self.num_units)
        self.comm_dropout = nn.Dropout(p=input_dropout)
        self.input_dropout = nn.Dropout(p=comm_dropout)

    def transpose_for_scores(self, x, num_attention_heads, attention_head_size):
        new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def input_attention_mask(self, x, h, option):
        """
        Input : x (batch_size, 2, input_size) [The null input is appended along the first dimension]
                h (batch_size, num_units, hidden_size)
        Output: inputs (list of size num_units with each element of shape (batch_size, input_value_size))
                mask_ binary array of shape (batch_size, num_units) where 1 indicates active and 0 indicates inactive
        """
        value_layer = self.value(x)
        value_layer = torch.mean(self.transpose_for_scores(value_layer, self.num_input_heads, self.input_value_size), dim=1)

        attention_scores = torch.einsum("bio, on  -> bni", option, self.p_w)

        if len(self.p_b.shape) == 1:
            p_b = self.p_b.unsqueeze(0)
            p_b = p_b.expand(self.batch_size, -1)

            null_input = torch.zeros(self.batch_size, attention_scores.shape[1]).float().to(self.device)
            p_b = torch.stack((p_b, null_input), dim=2)
            assert p_b.shape == (self.batch_size, self.num_units, 2)

        attention_scores = attention_scores + p_b

        assert value_layer.shape == (self.batch_size, 2, self.input_value_size)
        assert attention_scores.shape == (self.batch_size, self.num_units, 2)

        mask_ = torch.zeros(x.size(0), self.num_units).to(self.device)

        not_null_scores = attention_scores[:, :, 0]
        topk1 = torch.topk(not_null_scores, self.k, dim=1)
        row_index = np.arange(x.size(0))
        row_index = np.repeat(row_index, self.k)

        mask_[row_index, topk1.indices.view(-1)] = 1

        attention_probs = self.input_dropout(nn.Softmax(dim=-1)(attention_scores))
        inputs = torch.matmul(attention_probs, value_layer) * mask_.unsqueeze(2)

        assert inputs.shape == (self.batch_size, self.num_units, self.input_value_size)
        assert mask_.shape == (self.batch_size, self.num_units)

        return inputs, mask_

    def communication_attention(self, h, mask):
        """
        Input : h (batch_size, num_units, hidden_size)
                mask obtained from the input_attention_mask() function
        Output: context_layer (batch_size, num_units, hidden_size). New hidden states after communication
        """
        query_layer = self.query_(h)
        key_layer = self.key_(h)
        value_layer = self.value_(h)

        assert query_layer.shape == (self.batch_size, self.num_units, self.comm_query_size * self.num_comm_heads)
        assert key_layer.shape == (self.batch_size, self.num_units, self.comm_key_size * self.num_comm_heads)
        assert value_layer.shape == (self.batch_size, self.num_units, self.comm_value_size * self.num_comm_heads)

        query_layer = self.transpose_for_scores(query_layer, self.num_comm_heads, self.comm_query_size)
        key_layer = self.transpose_for_scores(key_layer, self.num_comm_heads, self.comm_key_size)
        value_layer = self.transpose_for_scores(value_layer, self.num_comm_heads, self.comm_value_size)

        assert query_layer.shape == (self.batch_size, self.num_comm_heads, self.num_units, self.comm_query_size)
        assert key_layer.shape == (self.batch_size, self.num_comm_heads, self.num_units, self.comm_key_size)
        assert value_layer.shape == (self.batch_size, self.num_comm_heads, self.num_units, self.comm_value_size)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.comm_key_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        mask = [mask for _ in range(attention_probs.size(1))]
        mask = torch.stack(mask, dim=1)

        assert mask.shape == (self.batch_size, self.num_comm_heads, self.num_units)

        attention_probs = attention_probs * mask.unsqueeze(3)
        attention_probs = self.comm_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.num_comm_heads * self.comm_value_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = self.comm_attention_output(context_layer)
        context_layer = context_layer + h

        assert context_layer.shape == (self.batch_size, self.num_units, self.comm_value_size)
        return context_layer

    def forward(self, x, hs, option, cs=None):
        """
        Input : x (batch_size, 1 , input_size)
                hs (batch_size, num_units, hidden_size)
                cs (batch_size, num_units, hidden_size)
        Output: new hs, cs for LSTM
                new hs for GRU
        """
        size = x.size()
        self.batch_size = size[0]
        null_input = torch.zeros(size[0], 1, size[2]).float().to(self.device)
        x = torch.cat((x, null_input), dim=1)

        one_hot_option = torch.tensor(convert_onehot(option, self.option_num), dtype=torch.float32).unsqueeze(1)
        one_hot_option_size = one_hot_option.size()
        null_input_option = torch.zeros(one_hot_option_size[0], 1, one_hot_option_size[2]).float()
        one_hot_option = torch.cat((one_hot_option, null_input_option), dim=1)

        assert x.shape == (self.batch_size, 2, self.input_size)
        assert one_hot_option.shape == (self.batch_size, 2, self.option_num)

        assert hs.shape == (self.batch_size, self.num_units, self.hidden_size)
        assert cs.shape == (self.batch_size, self.num_units, self.hidden_size)

        # Compute input attention
        inputs, mask = self.input_attention_mask(x, hs, one_hot_option)
        h_old = hs * 1.0
        if cs is not None:
            c_old = cs * 1.0

        # Compute RNN(LSTM or GRU) output
        if cs is not None:
            hs, cs = self.rnn(inputs, (hs, cs))
        else:
            hs = self.rnn(inputs, hs)

        # Block gradient through inactive units
        mask = mask.unsqueeze(2)
        h_new = blocked_grad.apply(hs, mask)

        # Compute communication attention
        h_new = self.communication_attention(h_new, mask.squeeze(2))

        hs = mask * h_new + (1 - mask) * h_old
        if cs is not None:
            cs = mask * cs + (1 - mask) * c_old
            return hs, cs

        return hs, None
