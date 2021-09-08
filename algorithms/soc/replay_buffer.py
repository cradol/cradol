import torch
import gym
import numpy as np
from misc.torch_utils import combined_shape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBufferSOC(object):
    def __init__(self, env, size, hidden_size, rim_num, value_size):
        if isinstance(env.observation_space, gym.spaces.Discrete):
            raise NotImplementedError("It should be observation_space.dim. Note from DK")
        elif isinstance(env.observation_space, gym.spaces.Dict) and list(env.observation_space.spaces.keys()) == ["image"]:
            obs_dim = env.observation_space.spaces["image"].shape
        else:
            obs_dim = env.observation_space.shape[0]

        if isinstance(obs_dim, tuple):
            self.obs_buf = np.zeros((size, obs_dim[0], obs_dim[1], obs_dim[2]), dtype=np.float32)
            self.next_obs_buf = np.zeros((size, obs_dim[0], obs_dim[1], obs_dim[2]), dtype=np.float32)
        else:
            self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
            self.next_obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)

        action_dim = 1 if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0]

        self.factored_next_state_buf = np.zeros((size, value_size), dtype=np.float32)
        self.option_buf = np.zeros(combined_shape(size, 1), dtype=np.float32)
        self.action_buf = np.zeros(combined_shape(size, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)

        self.hidden_in_pi_buf = np.zeros((size, rim_num, hidden_size), dtype=np.float32)
        self.cell_in_pi_buf = np.zeros((size, rim_num, hidden_size), dtype=np.float32)

        self.hidden_in_beta_buf = np.zeros((size, hidden_size), dtype=np.float32)
        self.cell_in_beta_buf = np.zeros((size, hidden_size), dtype=np.float32)

        self.hidden_in_inter_q1_buf = np.zeros((size, hidden_size), dtype=np.float32)
        self.cell_in_inter_q1_buf = np.zeros((size, hidden_size), dtype=np.float32)
        self.hidden_out_inter_q1_buf = np.zeros((size, hidden_size), dtype=np.float32)
        self.cell_out_inter_q1_buf = np.zeros((size, hidden_size), dtype=np.float32)

        self.hidden_in_inter_q2_buf = np.zeros((size, hidden_size), dtype=np.float32)
        self.cell_in_inter_q2_buf = np.zeros((size, hidden_size), dtype=np.float32)
        self.hidden_out_inter_q2_buf = np.zeros((size, hidden_size), dtype=np.float32)
        self.cell_out_inter_q2_buf = np.zeros((size, hidden_size), dtype=np.float32)

        self.hidden_in_intra_q1_buf = np.zeros((size, hidden_size), dtype=np.float32)
        self.cell_in_intra_q1_buf = np.zeros((size, hidden_size), dtype=np.float32)
        self.hidden_out_intra_q1_buf = np.zeros((size, hidden_size), dtype=np.float32)
        self.cell_out_intra_q1_buf = np.zeros((size, hidden_size), dtype=np.float32)

        self.hidden_in_intra_q2_buf = np.zeros((size, hidden_size), dtype=np.float32)
        self.cell_in_intra_q2_buf = np.zeros((size, hidden_size), dtype=np.float32)
        self.hidden_out_intra_q2_buf = np.zeros((size, hidden_size), dtype=np.float32)
        self.cell_out_intra_q2_buf = np.zeros((size, hidden_size), dtype=np.float32)

        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, option, action, reward, next_obs, done,
              state_in_pi, state_in_beta,
              state_in_inter_q1, state_out_inter_q1,
              state_in_inter_q2, state_out_inter_q2,
              state_in_intra_q1, state_out_intra_q1,
              state_in_intra_q2, state_out_intra_q2, factored_next_state):

        self.obs_buf[self.ptr] = obs
        self.option_buf[self.ptr] = option
        self.next_obs_buf[self.ptr] = next_obs
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done

        self.hidden_in_pi_buf[self.ptr] = state_in_pi[0].cpu().detach().numpy()
        self.cell_in_pi_buf[self.ptr] = state_in_pi[1].cpu().detach().numpy()

        self.hidden_in_beta_buf[self.ptr] = state_in_beta[0].cpu().detach().numpy()
        self.cell_in_beta_buf[self.ptr] = state_in_beta[1].cpu().detach().numpy()

        self.hidden_in_inter_q1_buf[self.ptr] = state_in_inter_q1[0].cpu().detach().numpy()
        self.cell_in_inter_q1_buf[self.ptr] = state_in_inter_q1[1].cpu().detach().numpy()
        self.hidden_out_inter_q1_buf[self.ptr] = state_out_inter_q1[0].cpu().detach().numpy()
        self.cell_out_inter_q1_buf[self.ptr] = state_out_inter_q1[1].cpu().detach().numpy()

        self.hidden_in_inter_q2_buf[self.ptr] = state_in_inter_q2[0].cpu().detach().numpy()
        self.cell_in_inter_q2_buf[self.ptr] = state_in_inter_q2[1].cpu().detach().numpy()
        self.hidden_out_inter_q2_buf[self.ptr] = state_out_inter_q2[0].cpu().detach().numpy()
        self.cell_out_inter_q2_buf[self.ptr] = state_out_inter_q2[1].cpu().detach().numpy()

        self.hidden_in_intra_q1_buf[self.ptr] = state_in_intra_q1[0].cpu().detach().numpy()
        self.cell_in_intra_q1_buf[self.ptr] = state_in_intra_q1[1].cpu().detach().numpy()
        self.hidden_out_intra_q1_buf[self.ptr] = state_out_intra_q1[0].cpu().detach().numpy()
        self.cell_out_intra_q1_buf[self.ptr] = state_out_intra_q1[1].cpu().detach().numpy()

        self.hidden_in_intra_q2_buf[self.ptr] = state_in_intra_q2[0].cpu().detach().numpy()
        self.cell_in_intra_q2_buf[self.ptr] = state_in_intra_q2[1].cpu().detach().numpy()
        self.hidden_out_intra_q2_buf[self.ptr] = state_out_intra_q2[0].cpu().detach().numpy()
        self.cell_out_intra_q2_buf[self.ptr] = state_out_intra_q2[1].cpu().detach().numpy()

        self.factored_next_state_buf[self.ptr] = factored_next_state.cpu().detach().numpy()

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            option=self.option_buf[idxs],
            action=self.action_buf[idxs],
            reward=self.reward_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            done=self.done_buf[idxs],
            hidden_in_pi=self.hidden_in_pi_buf[idxs],
            cell_in_pi=self.cell_in_pi_buf[idxs],
            hidden_in_beta=self.hidden_in_beta_buf[idxs],
            cell_in_beta=self.cell_in_beta_buf[idxs],
            hidden_in_inter_q1=self.hidden_in_inter_q1_buf[idxs],
            cell_in_inter_q1=self.cell_in_inter_q1_buf[idxs],
            hidden_out_inter_q1=self.hidden_out_inter_q1_buf[idxs],
            cell_out_inter_q1=self.cell_out_inter_q1_buf[idxs],
            hidden_in_inter_q2=self.hidden_in_inter_q2_buf[idxs],
            cell_in_inter_q2=self.cell_in_inter_q2_buf[idxs],
            hidden_out_inter_q2=self.hidden_out_inter_q2_buf[idxs],
            cell_out_inter_q2=self.cell_out_inter_q2_buf[idxs],
            hidden_in_intra_q1=self.hidden_in_intra_q1_buf[idxs],
            cell_in_intra_q1=self.cell_in_intra_q1_buf[idxs],
            hidden_out_intra_q1=self.hidden_out_intra_q1_buf[idxs],
            cell_out_intra_q1=self.cell_out_intra_q1_buf[idxs],
            hidden_in_intra_q2=self.hidden_in_intra_q2_buf[idxs],
            cell_in_intra_q2=self.cell_in_intra_q2_buf[idxs],
            hidden_out_intra_q2=self.hidden_out_intra_q2_buf[idxs],
            cell_out_intra_q2=self.cell_out_intra_q2_buf[idxs],
            factored_next_state=self.factored_next_state_buf[idxs]
        )

        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in batch.items()}
