import itertools
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.optim import Adam
from torch.distributions.categorical import Categorical
from copy import deepcopy
from misc.torch_utils import convert_onehot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SoftOptionCritic(nn.Module):
    def __init__(self, env, args, tb_writer, log):
        super(SoftOptionCritic, self).__init__()

        self.action_space = env.action_space
        self.args = args
        self.tb_writer = tb_writer
        self.log = log

        # Define obs_dim
        if isinstance(env.observation_space, gym.spaces.Discrete):
            raise NotImplementedError("It should be observation_space.dim. Note from DK")
        elif isinstance(env.observation_space, gym.spaces.Dict) and list(env.observation_space.spaces.keys()) == ["image"]:
            self.obs_dim = env.observation_space.spaces["image"].shape
        else:
            self.obs_dim = env.observation_space.shape[0]

        log[args.log_name].info("Observation dim: {}".format(self.obs_dim))

        # Define model and target model
        if isinstance(self.action_space, gym.spaces.Discrete):
            if isinstance(self.obs_dim, tuple):
                from algorithms.soc.discrete_model_image import SOCModelImageCategorical
                self.model = SOCModelImageCategorical(
                    self.obs_dim, self.action_space.n, args.hidden_size, args.option_num,
                    args.rim_num, args.k, args.value_size).to(device)
            else:
                from algorithms.soc.discrete_model import SOCModelCategorical
                self.model = SOCModelCategorical(
                    self.obs_dim, self.action_space.n, args.hidden_size, args.option_num,
                    args.rim_num, args.k, args.value_size).to(device)
        else:
            from algorithms.soc.continuous_model import SOCModelContinous
            self.model = SOCModelContinous(
                self.obs_dim, self.action_space.shape[0], args.hidden_size, args.option_num,
                args.rim_num, args.k, args.value_size, self.action_space.high[0]).to(device)

        self.model_target = deepcopy(self.model)

        # Parameter Definitions
        self.model.total_q_params = itertools.chain(
            self.model.inter_q_function_1.parameters(),
            self.model.inter_q_function_2.parameters(),
            self.model.intra_q_function_1.parameters(),
            self.model.intra_q_function_2.parameters())

        # Define optimizers
        self.intra_policy_optim = Adam(self.model.intra_option_policy.parameters(), lr=args.lr)
        self.beta_optim = Adam(self.model.beta_policy.parameters(), lr=args.lr)
        self.q_function_optim = Adam(self.model.total_q_params, lr=args.lr)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.model_target.parameters():
            p.requires_grad = False

        self.iteration = 0

    def factored_input(self, x, option, target=False):
        if target:
            rims_model = self.model_target.intra_option_policy.pi_rnn
            if isinstance(self.obs_dim, tuple):
                image_conv = self.model_target.intra_option_policy.pi_pre_fc
            else:
                pre_fc = self.model_target.intra_option_policy.pi_pre_fc

        else:
            rims_model = self.model.intra_option_policy.pi_rnn
            if isinstance(self.obs_dim, tuple):
                image_conv = self.model.intra_option_policy.pi_pre_fc
            else:
                pre_fc = self.model_target.intra_option_policy.pi_pre_fc

        if isinstance(self.obs_dim, tuple):
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
            if len(option.shape) == 1:
                option = option.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            elif len(option.shape) == 2:
                option = option.unsqueeze(0).unsqueeze(0)

            x = x.transpose(1, 3).transpose(2, 3)
            x = image_conv(x)

            x = x.reshape(x.shape[0], -1)
        else:
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            if len(x.shape) == 2:
                x = x.unsqueeze(1)

            x = pre_fc(x)

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        size = x.size()
        batch_size = size[0]
        null_input = torch.zeros(size[0], 1, size[2]).float().to(device)
        x = torch.cat((x, null_input), dim=1)

        one_hot_option = torch.tensor(convert_onehot(option, rims_model.option_num), dtype=torch.float32).unsqueeze(1)
        one_hot_option_size = one_hot_option.size()
        null_input_option = torch.zeros(one_hot_option_size[0], 1, one_hot_option_size[2]).float()
        one_hot_option = torch.cat((one_hot_option, null_input_option), dim=1)

        assert x.shape == (batch_size, 2, rims_model.input_size)
        assert one_hot_option.shape == (rims_model.batch_size, 2, rims_model.option_num)

        # Compute input attention
        value_layer = rims_model.value(x)
        value_layer = torch.mean(rims_model.transpose_for_scores(
            value_layer, rims_model.num_input_heads, rims_model.input_value_size), dim=1)

        attention_scores = torch.einsum("bio, on  -> bni", one_hot_option, rims_model.p_w)

        if len(rims_model.p_b.shape) == 1:
            p_b = rims_model.p_b.unsqueeze(0)
            p_b = p_b.expand(batch_size, -1)

            null_input = torch.zeros(batch_size, attention_scores.shape[1]).float().to(device)
            p_b = torch.stack((p_b, null_input), dim=2)
            assert p_b.shape == (batch_size, rims_model.num_units, 2)

        attention_scores = attention_scores + p_b

        assert value_layer.shape == (batch_size, 2, rims_model.input_value_size)
        assert attention_scores.shape == (batch_size, rims_model.num_units, 2)

        mask_ = torch.zeros(x.size(0), rims_model.num_units).to(device)

        not_null_scores = attention_scores[:, :, 0]
        topk1 = torch.topk(not_null_scores, rims_model.k, dim=1)

        row_index = np.arange(x.size(0))
        row_index = np.repeat(row_index, rims_model.k)

        mask_[row_index, topk1.indices.view(-1)] = 1
        attention_probs = rims_model.input_dropout(nn.Softmax(dim=-1)(attention_scores))

        inputs = torch.matmul(attention_probs, value_layer) * mask_.unsqueeze(2)
        inputs = torch.mean(inputs, dim=1)

        return inputs.squeeze(0)

    def visualization_output(self, x, option, start_in_pi, target=False, communication=True):
        x = torch.FloatTensor(x)
        option = torch.tensor(option)

        if target:
            rims_model = self.model_target.intra_option_policy.pi_rnn
            if isinstance(self.obs_dim, tuple):
                image_conv = self.model_target.intra_option_policy.pi_pre_fc
            else:
                pre_fc = self.model_target.intra_option_policy.pi_pre_fc

        else:
            rims_model = self.model.intra_option_policy.pi_rnn
            if isinstance(self.obs_dim, tuple):
                image_conv = self.model.intra_option_policy.pi_pre_fc
            else:
                pre_fc = self.model_target.intra_option_policy.pi_pre_fc

        if isinstance(self.obs_dim, tuple):
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
            if len(option.shape) == 1:
                option = option.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            elif len(option.shape) == 2:
                option = option.unsqueeze(0).unsqueeze(0)

            x = x.transpose(1, 3).transpose(2, 3)
            x = image_conv(x)

            x = x.reshape(x.shape[0], -1)
        else:
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            if len(x.shape) == 2:
                x = x.unsqueeze(1)

            x = pre_fc(x)

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        size = x.size()
        batch_size = size[0]
        null_input = torch.zeros(size[0], 1, size[2]).float().to(device)
        x = torch.cat((x, null_input), dim=1)

        one_hot_option = torch.tensor(convert_onehot(option, rims_model.option_num), dtype=torch.float32).unsqueeze(1)
        one_hot_option_size = one_hot_option.size()
        null_input_option = torch.zeros(one_hot_option_size[0], 1, one_hot_option_size[2]).float()
        one_hot_option = torch.cat((one_hot_option, null_input_option), dim=1)

        assert x.shape == (batch_size, 2, rims_model.input_size)
        assert one_hot_option.shape == (rims_model.batch_size, 2, rims_model.option_num)

        # Compute input attention
        value_layer = rims_model.value(x)
        value_layer = torch.mean(
            rims_model.transpose_for_scores(value_layer, rims_model.num_input_heads, rims_model.input_value_size),
            dim=1)

        attention_scores = torch.einsum("bio, on  -> bni", one_hot_option, rims_model.p_w)

        if len(rims_model.p_b.shape) == 1:
            p_b = rims_model.p_b.unsqueeze(0)
            p_b = p_b.expand(batch_size, -1)

            null_input = torch.zeros(batch_size, attention_scores.shape[1]).float().to(device)
            p_b = torch.stack((p_b, null_input), dim=2)
            assert p_b.shape == (batch_size, rims_model.num_units, 2)

        attention_scores = attention_scores + p_b

        assert value_layer.shape == (batch_size, 2, rims_model.input_value_size)
        assert attention_scores.shape == (batch_size, rims_model.num_units, 2)

        mask_ = torch.zeros(x.size(0), rims_model.num_units).to(device)

        not_null_scores = attention_scores[:, :, 0]
        topk1 = torch.topk(not_null_scores, rims_model.k, dim=1)

        row_index = np.arange(x.size(0))
        row_index = np.repeat(row_index, rims_model.k)

        mask_[row_index, topk1.indices.view(-1)] = 1

        attention_probs = rims_model.input_dropout(nn.Softmax(dim=-1)(attention_scores))

        inputs = torch.matmul(attention_probs, value_layer) * mask_.unsqueeze(2)
        rims_model = self.model.intra_option_policy.pi_rnn

        assert inputs.shape == (rims_model.batch_size, rims_model.num_units, rims_model.input_value_size)
        assert mask_.shape == (rims_model.batch_size, rims_model.num_units)

        hs = start_in_pi[0]
        cs = start_in_pi[1]

        if len(hs.shape) == 2:
            hs = hs.unsqueeze(0)
            cs = cs.unsqueeze(0)

        h_old = hs * 1.0
        if cs is not None:
            c_old = cs * 1.0

        hs, cs = rims_model.rnn(inputs, (hs, cs))

        if communication:
            from algorithms.soc.RIM import blocked_grad
            mask = mask_.unsqueeze(2)
            h_new = blocked_grad.apply(hs, mask)

            # Compute communication attention
            h_new = rims_model.communication_attention(h_new, mask.squeeze(2))

            hs = mask * h_new + (1 - mask) * h_old
            cs = mask * cs + (1 - mask) * c_old
            hs = torch.mean(hs, dim=1).squeeze(0)
        else:
            hs = torch.mean(hs, dim=1).squeeze(0)

        return hs

    def get_option(self, state1, state2):
        q1, state_out_inter_q1 = self.model.inter_q_function_1(state1)
        q2, state_out_inter_q2 = self.model.inter_q_function_2(state2)
        q = torch.min(q1, q2)

        if self.args.option_num != 1:
            q_distribution = Categorical(logits=q)
            option = q_distribution.sample()
        else:
            if len(q1.shape) == 1:
                option = torch.from_numpy(np.array([0], dtype=np.int64)).squeeze(-1)
            elif len(q1.shape) == 2:
                batch_size = q1.shape[0]
                option = torch.from_numpy(np.zeros((batch_size, 1), dtype=np.int64))
            else:
                raise ValueError()

        return option, state_out_inter_q1, state_out_inter_q2

    def predict_option_termination(self, state, option_indices):
        one_hot_option = torch.FloatTensor(convert_onehot(option_indices, self.args.option_num)).squeeze(0)

        termination, state_out_beta = self.model.beta_policy(state, one_hot_option)
        option_termination = Bernoulli(termination).sample()

        return termination, option_termination == 1, state_out_beta

    def get_action(self, state, option_indices):
        if isinstance(self.action_space, gym.spaces.Discrete):
            # Get action
            action_probs, state_out_pi = self.model.intra_option_policy(state, option_indices)
            action_distribution = Categorical(probs=action_probs)
            action = action_distribution.sample()
            option_indices = option_indices.clone().detach().long().unsqueeze(-1).unsqueeze(-1)
            assert len(action.shape) == len(option_indices.shape), "{} vs {}".format(action.shape, option_indices.shape)

            action = torch.gather(action, 1, option_indices).squeeze(-1)

            # Get logp
            z = (action_probs == 0.).float() * 1e-8
            logp = torch.log(action_probs + z)

            return action.cpu().numpy()[0], logp, state_out_pi
        else:
            action, logp, state_out_pi = self.model.intra_option_policy(state, option_indices)
            action = action.squeeze(0)[option_indices, :]
            return action.detach().cpu().numpy(), logp, state_out_pi

    def get_action_probs(self, state, option_indices):
        if isinstance(self.action_space, gym.spaces.Discrete):
            action_probs, state_out_pi = self.model.intra_option_policy(state, option_indices)
            assert action_probs.shape == (self.args.batch_size, self.args.option_num, self.action_space.n)
            z = (action_probs == 0.).float() * 1e-8
            logp = torch.log(action_probs + z)
            assert logp.shape == (self.args.batch_size, self.args.option_num, self.action_space.n)

            action_probs = action_probs[torch.arange(self.args.batch_size), option_indices.squeeze(), :]
            logp = logp[torch.arange(self.args.batch_size), option_indices.squeeze(), :]

            assert action_probs.shape == (self.args.batch_size, self.action_space.n)
            assert logp.shape == (self.args.batch_size, self.action_space.n)
        else:
            action_probs, logp, state_out_pi = self.model.intra_option_policy(state, option_indices)
            assert action_probs.shape == (self.args.batch_size, self.args.option_num, self.action_space.shape[0])
            action_probs = action_probs[torch.arange(self.args.batch_size), option_indices.squeeze(), :]
            logp = torch.gather(logp, 1, option_indices)

            assert action_probs.shape == (self.args.batch_size, self.action_space.shape[0])
            assert logp.shape == (self.args.batch_size, 1)

        return action_probs, logp

    def compute_loss_beta(self, next_state, option_indices, done,
                          state_in_beta, state_out_inter_q1, state_out_inter_q2):

        factored_next_state = self.factored_input(next_state, option_indices)
        beta_prob, _, _ = self.predict_option_termination((factored_next_state, state_in_beta), option_indices)
        assert beta_prob.shape == (self.args.batch_size, 1)

        # Computing Inter Q Values for Advantage
        with torch.no_grad():
            total_next_state_inter_q1 = (next_state, state_out_inter_q1)
            total_next_state_inter_q2 = (next_state, state_out_inter_q2)

            q1_pi, _ = self.model.inter_q_function_1(total_next_state_inter_q1)
            q2_pi, _ = self.model.inter_q_function_2(total_next_state_inter_q2)
            q_pi = torch.min(q1_pi, q2_pi)

            assert q_pi.shape == (self.args.batch_size, self.args.option_num)

            q_pi_current_option = torch.gather(q_pi, 1, option_indices)
            q_pi_next_option = torch.max(q_pi, dim=1)[0].unsqueeze(-1)

            assert q_pi_current_option.shape == (self.args.batch_size, 1)
            assert q_pi_next_option.shape == (self.args.batch_size, 1)

            advantage = q_pi_current_option - q_pi_next_option
            assert advantage.shape == (self.args.batch_size, 1)
            assert torch.logical_not(done).shape == (self.args.batch_size, 1)

        # Beta Policy Loss
        loss_beta = ((beta_prob * advantage) * torch.logical_not(done)).mean()

        return loss_beta

    def compute_loss_inter(self, state, option_indices, one_hot_option, action_probs, logp,
                           state_in_inter_q1, state_in_inter_q2, state_in_intra_q1, state_in_intra_q2):

        # Computer Inter-Q Values
        total_state_inter_q1 = (state, state_in_inter_q1)
        total_state_inter_q2 = (state, state_in_inter_q2)

        q1_inter_all, _ = self.model.inter_q_function_1(total_state_inter_q1)
        q2_inter_all, _ = self.model.inter_q_function_2(total_state_inter_q2)

        assert q1_inter_all.shape == (self.args.batch_size, self.args.option_num)

        q1_inter = torch.gather(q1_inter_all, 1, option_indices)
        q2_inter = torch.gather(q2_inter_all, 1, option_indices)
        assert q1_inter.shape == (self.args.batch_size, 1)
        assert q2_inter.shape == (self.args.batch_size, 1)

        # Intra Q Values for Target
        with torch.no_grad():
            if isinstance(self.action_space, gym.spaces.Discrete):
                # Hidden State Intra Pi from Replay Buffer
                q1_intra_targ, _ = self.model_target.intra_q_function_1((state, state_in_intra_q1), one_hot_option)
                q2_intra_targ, _ = self.model_target.intra_q_function_2((state, state_in_intra_q2), one_hot_option)
            else:
                q1_intra_targ, _ = self.model_target.intra_q_function_1(
                    (state, state_in_intra_q1), one_hot_option, action_probs)
                q2_intra_targ, _ = self.model_target.intra_q_function_2(
                    (state, state_in_intra_q2), one_hot_option, action_probs)
            q_intra_targ = torch.min(q1_intra_targ, q2_intra_targ)

            # Inter-Q Back Up Values
            if isinstance(self.action_space, gym.spaces.Discrete):
                assert q_intra_targ.shape == (self.args.batch_size, self.action_space.n)
                backup_inter = (action_probs * (q_intra_targ - (self.args.alpha * logp))).sum(dim=-1)
                backup_inter = torch.unsqueeze(backup_inter, -1)
            else:
                assert q_intra_targ.shape == (self.args.batch_size, 1)
                backup_inter = q_intra_targ - (self.args.alpha * logp)

            assert backup_inter.shape == (self.args.batch_size, 1)

        # Inter-Q Function Loss
        loss_inter_q1 = F.mse_loss(q1_inter, backup_inter)
        loss_inter_q2 = F.mse_loss(q2_inter, backup_inter)
        loss_inter_q = loss_inter_q1 + loss_inter_q2

        entropy_debug = (-action_probs * logp).sum(dim=-1).mean().detach()
        self.tb_writer.log_data("loss/entropy", self.iteration, entropy_debug)

        return loss_inter_q

    def compute_loss_intra_q(self, state, action, one_hot_option, option_indices, next_state, reward, done,
                             state_in_beta, state_out_inter_q1, state_out_inter_q2,
                             state_in_intra_q1, state_in_intra_q2):

        if isinstance(self.action_space, gym.spaces.Discrete):
            # Hidden State Intra Pi from Replay Buffer
            q1_intra, _ = self.model.intra_q_function_1((state, state_in_intra_q1), one_hot_option)
            q2_intra, _ = self.model.intra_q_function_2((state, state_in_intra_q2), one_hot_option)
            assert q1_intra.shape == (self.args.batch_size, self.action_space.n)
        else:
            q1_intra, _ = self.model.intra_q_function_1((state, state_in_intra_q1), one_hot_option, action)
            q2_intra, _ = self.model.intra_q_function_2((state, state_in_intra_q2), one_hot_option, action)
            assert q1_intra.shape == (self.args.batch_size, 1)

        # Beta Prob for Target
        factored_next_state = self.factored_input(next_state, option_indices)
        beta_prob, _, _ = self.predict_option_termination((factored_next_state, state_in_beta), option_indices)
        assert beta_prob.shape == (self.args.batch_size, 1)

        # Inter Q Values for Target
        with torch.no_grad():
            total_next_state_inter_q1 = (next_state, state_out_inter_q1)
            total_next_state_inter_q2 = (next_state, state_out_inter_q2)

            q1_inter_targ, _ = self.model_target.inter_q_function_1(total_next_state_inter_q1)
            q2_inter_targ, _ = self.model_target.inter_q_function_2(total_next_state_inter_q2)
            q_inter_targ = torch.min(q1_inter_targ, q2_inter_targ)

            assert q_inter_targ.shape == (self.args.batch_size, self.args.option_num)

            q_inter_targ_current_option = torch.gather(q_inter_targ, 1, option_indices)
            assert q_inter_targ_current_option.shape == (self.args.batch_size, 1)

            next_option, _, _ = self.get_option(total_next_state_inter_q1, total_next_state_inter_q2)

            if len(next_option.shape) == 1:
                next_option = next_option.unsqueeze(-1)
            assert next_option.shape == (self.args.batch_size, 1)

            q_inter_targ_next_option = torch.gather(q_inter_targ, 1, next_option)
            assert q_inter_targ_next_option.shape == (self.args.batch_size, 1)

            # Intra-Q Back Up Values
            backup_intra = reward + self.args.gamma * torch.logical_not(done) * (
                ((1. - beta_prob) * q_inter_targ_current_option) + (beta_prob * q_inter_targ_next_option))
            assert backup_intra.shape == (self.args.batch_size, 1)

        # Computing Intra Q Loss
        if isinstance(self.action_space, gym.spaces.Discrete):
            loss_intra_q1 = F.mse_loss(torch.gather(q1_intra, 1, action.long()), backup_intra.detach())
            loss_intra_q2 = F.mse_loss(torch.gather(q2_intra, 1, action.long()), backup_intra.detach())
        else:
            loss_intra_q1 = F.mse_loss(q1_intra, backup_intra.detach())
            loss_intra_q2 = F.mse_loss(q2_intra, backup_intra.detach())

        loss_intra_q = loss_intra_q1 + loss_intra_q2

        return loss_intra_q

    def compute_loss_intra_policy(self, state, action_probs, logp, one_hot_option,
                                  state_in_intra_q1, state_in_intra_q2):
        if isinstance(self.action_space, gym.spaces.Discrete):
            # Online Hidden State from Intra Pi for Intra Q
            q1_intra_current_action, _ = self.model.intra_q_function_1(
                (state, state_in_intra_q1), one_hot_option)
            q2_intra_current_action, _ = self.model.intra_q_function_2(
                (state, state_in_intra_q2), one_hot_option)
        else:
            q1_intra_current_action, _ = self.model.intra_q_function_1(
                (state, state_in_intra_q1), one_hot_option, action_probs)
            q2_intra_current_action, _ = self.model.intra_q_function_2(
                (state, state_in_intra_q2), one_hot_option, action_probs)
        q_pi = torch.min(q1_intra_current_action, q2_intra_current_action)

        # Intra-Policy Loss
        if isinstance(self.action_space, gym.spaces.Discrete):
            assert q_pi.shape == (self.args.batch_size, self.action_space.n)
            loss_intra_pi = (action_probs * (self.args.alpha * logp - q_pi)).sum(dim=1).mean()
        else:
            assert q_pi.shape == (self.args.batch_size, 1)
            loss_intra_pi = (self.args.alpha * logp - q_pi).mean()

        return loss_intra_pi

    def update_loss_soc(self, data):
        # Parse data
        obs, option, action, reward, next_obs, done = \
            data['obs'], data['option'], data['action'], data['reward'], data['next_obs'], data['done']
        reward, done = reward.unsqueeze(-1), done.unsqueeze(-1)

        hidden_in_pi, cell_in_pi = data['hidden_in_pi'], data['cell_in_pi']
        hidden_in_beta, cell_in_beta = data['hidden_in_beta'], data['cell_in_beta']
        hidden_in_inter_q1, cell_in_inter_q1 = data['hidden_in_inter_q1'], data['cell_in_inter_q1']
        hidden_out_inter_q1, cell_out_inter_q1 = data['hidden_out_inter_q1'], data['cell_out_inter_q1']
        hidden_in_inter_q2, cell_in_inter_q2 = data['hidden_in_inter_q2'], data['cell_in_inter_q2']
        hidden_out_inter_q2, cell_out_inter_q2 = data['hidden_out_inter_q2'], data['cell_out_inter_q2']
        hidden_in_intra_q1, cell_in_intra_q1 = data['hidden_in_intra_q1'], data['cell_in_intra_q1']
        hidden_in_intra_q2, cell_in_intra_q2 = data['hidden_in_intra_q2'], data['cell_in_intra_q2']

        if self.args.env_name.find("MiniGrid") != -1:
            assert obs.shape == (self.args.batch_size, self.obs_dim[0], self.obs_dim[1], self.obs_dim[2])
            assert next_obs.shape == (self.args.batch_size, self.obs_dim[0], self.obs_dim[1], self.obs_dim[2])
        else:
            assert obs.shape == (self.args.batch_size, self.obs_dim)
            assert next_obs.shape == (self.args.batch_size, self.obs_dim)

        assert option.shape == (self.args.batch_size, 1)
        assert reward.shape == (self.args.batch_size, 1)
        assert done.shape == (self.args.batch_size, 1)
        assert hidden_in_pi.shape == (self.args.batch_size, self.args.rim_num, self.args.hidden_size)
        assert cell_in_pi.shape == (self.args.batch_size, self.args.rim_num, self.args.hidden_size)
        if isinstance(self.action_space, gym.spaces.Discrete):
            assert action.shape == (self.args.batch_size, 1)
        else:
            assert action.shape == (self.args.batch_size, self.action_space.shape[0])

        # Process option
        one_hot_option = torch.tensor(convert_onehot(option, self.args.option_num), dtype=torch.float32)
        option_indices = torch.LongTensor(option.cpu().numpy().flatten().astype(int)).unsqueeze(-1).to(device)
        assert one_hot_option.shape == (self.args.batch_size, self.args.option_num)
        assert option_indices.shape == (self.args.batch_size, 1)

        # Compute action probs and logp
        total_state_pi = (obs, (hidden_in_pi, cell_in_pi))
        action_probs, logp = self.get_action_probs(total_state_pi, option_indices)

        # Update q networks
        loss_intra_q = self.compute_loss_intra_q(
            obs, action, one_hot_option, option_indices, next_obs, reward, done,
            (hidden_in_beta, cell_in_beta),
            (hidden_out_inter_q1, cell_out_inter_q1),
            (hidden_out_inter_q2, cell_out_inter_q2),
            (hidden_in_intra_q1, cell_in_intra_q1),
            (hidden_in_intra_q2, cell_in_intra_q2))
        self.tb_writer.log_data("loss/q_function_loss/intra", self.iteration, loss_intra_q.item())

        # Calculate inter-q loss
        loss_inter_q = self.compute_loss_inter(
            obs, option_indices, one_hot_option, action_probs, logp,
            (hidden_in_inter_q1, cell_in_inter_q1),
            (hidden_in_inter_q2, cell_in_inter_q2),
            (hidden_in_intra_q1, cell_in_intra_q1),
            (hidden_in_intra_q2, cell_in_intra_q2))
        self.tb_writer.log_data("loss/q_function_loss/inter", self.iteration, loss_inter_q.item())

        total_loss = (loss_inter_q + loss_intra_q)
        self.q_function_optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.total_q_params, self.args.max_grad_clip)
        self.q_function_optim.step()

        # Update intra-policy
        loss_intra_pi = self.compute_loss_intra_policy(
            obs, action_probs, logp, one_hot_option,
            (hidden_in_intra_q1, cell_in_intra_q1),
            (hidden_in_intra_q2, cell_in_intra_q2))

        self.intra_policy_optim.zero_grad()
        loss_intra_pi.backward()
        torch.nn.utils.clip_grad_norm_(self.model.intra_option_policy.parameters(), self.args.max_grad_clip)
        self.intra_policy_optim.step()
        self.tb_writer.log_data("loss/intra_q_policy_loss", self.iteration, loss_intra_pi.item())

        # Update beta
        loss_beta = self.compute_loss_beta(
            next_obs, option_indices, done,
            (hidden_in_beta, cell_in_beta),
            (hidden_out_inter_q1, cell_out_inter_q1),
            (hidden_out_inter_q2, cell_out_inter_q2))

        self.beta_optim.zero_grad()
        loss_beta.backward()
        torch.nn.utils.clip_grad_norm_(self.model.beta_policy.parameters(), self.args.max_grad_clip)
        self.beta_optim.step()
        self.tb_writer.log_data("loss/beta_policy_loss", self.iteration, loss_beta.item())

        with torch.no_grad():
            for p, p_targ in zip(self.model.inter_q_function_1.parameters(),
                                 self.model_target.inter_q_function_1.parameters()):
                p_targ.data.mul_(self.args.polyak)
                p_targ.data.add_((1 - self.args.polyak) * p.data)

            for p, p_targ in zip(self.model.inter_q_function_2.parameters(),
                                 self.model_target.inter_q_function_2.parameters()):
                p_targ.data.mul_(self.args.polyak)
                p_targ.data.add_((1 - self.args.polyak) * p.data)

            for p, p_targ in zip(self.model.intra_q_function_1.parameters(),
                                 self.model_target.intra_q_function_1.parameters()):
                p_targ.data.mul_(self.args.polyak)
                p_targ.data.add_((1 - self.args.polyak) * p.data)

            for p, p_targ in zip(self.model.intra_q_function_2.parameters(),
                                 self.model_target.intra_q_function_2.parameters()):
                p_targ.data.mul_(self.args.polyak)
                p_targ.data.add_((1 - self.args.polyak) * p.data)

        self.iteration += 1

    def update_state_q(self, obs, target):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)

        total_state_q1 = obs
        total_state_q2 = obs

        if target:
            state_out_q1 = self.model_target.shared_model_1(total_state_q1)
            state_out_q2 = self.model_target.shared_model_2(total_state_q2)
        else:
            state_out_q1 = self.model.shared_model_1(total_state_q1)
            state_out_q2 = self.model.shared_model_2(total_state_q2)

        assert state_out_q1.shape == state_out_q2.shape

        return state_out_q1, state_out_q2

    def load_model(self, model_dir):
        self.model.load_state_dict(torch.load(model_dir, map_location=lambda storage, loc: storage))
