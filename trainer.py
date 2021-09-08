import torch
import gym
import random
from misc.utils import initialize_lstm_state, process_done_signal
from misc.torch_utils import convert_onehot


def train(agent, env, replay_buffer, args):
    # Initialization LSTM States
    state_in_pi, state_in_beta, state_in_inter_q1, state_in_inter_q2, state_in_intra_q1, state_in_intra_q2 = \
        initialize_lstm_state(args)

    # Initialization of Episode
    obs, ep_reward, ep_len, beta = env.reset(), 0, 0, True
    if args.env_name.find("MiniGrid") != -1:
        obs = obs['image']

    option_dict = dict((zip([str(x) for x in range(args.option_num)], [0] * args.option_num)))

    for total_step_count in range(args.total_step_num):
        if beta:
            # Get a new option when beta is True
            total_state_inter_q1 = (torch.FloatTensor(obs), state_in_inter_q1)
            total_state_inter_q2 = (torch.FloatTensor(obs), state_in_inter_q2)
            agent.current_option, state_out_inter_q1, state_out_inter_q2 = agent.get_option(
                total_state_inter_q1, total_state_inter_q2)
            if total_step_count < args.start_steps:
                agent.current_option = torch.tensor(random.randint(0, args.option_num - 1))
        else:
            # If not, simply update LSTM states for inter Q
            _, state_out_inter_q1, state_out_inter_q2 = agent.get_option(
                total_state_inter_q1, total_state_inter_q2)

        # Get primitive action
        option_dict[str(agent.current_option.item())] += 1

        total_state_pi = (torch.FloatTensor(obs), state_in_pi)
        action, _, state_out_pi = agent.get_action(total_state_pi, agent.current_option)

        # Update LSTM states for intra Q
        total_state_intra_q1 = (torch.FloatTensor(obs), state_in_intra_q1)
        total_state_intra_q2 = (torch.FloatTensor(obs), state_in_intra_q2)
        one_hot_option = torch.FloatTensor(convert_onehot(agent.current_option, args.option_num)).squeeze(0)

        if isinstance(agent.action_space, gym.spaces.Discrete):
            _, state_out_intra_q1 = agent.model.intra_q_function_1(total_state_intra_q1, one_hot_option)
            _, state_out_intra_q2 = agent.model.intra_q_function_2(total_state_intra_q2, one_hot_option)
        else:
            action = torch.FloatTensor(action)
            _, state_out_intra_q1 = agent.model.intra_q_function_1(total_state_intra_q1, one_hot_option, action)
            _, state_out_intra_q2 = agent.model.intra_q_function_2(total_state_intra_q2, one_hot_option, action)

        # Take action in the environment
        if total_step_count < args.start_steps:
            action = random.randint(0, env.action_space.n - 1)

        next_obs, reward, done, info = env.step(action)

        if args.env_name.find("MiniGrid") != -1:
            next_obs = next_obs['image']
        ep_len += 1
        ep_reward += reward
        done = process_done_signal(env, ep_len, reward, done, args)

        # Get beta
        factored_next_state = agent.factored_input(torch.FloatTensor(next_obs), agent.current_option)
        total_state_beta = (factored_next_state, state_in_beta)
        _, beta, state_out_beta = agent.predict_option_termination(total_state_beta, agent.current_option)

        # Store replay buffer
        replay_buffer.store(
            obs, agent.current_option, action, reward, next_obs, done,
            state_in_pi, state_in_beta,
            state_in_inter_q1, state_out_inter_q1,
            state_in_inter_q2, state_out_inter_q2,
            state_in_intra_q1, state_out_intra_q1,
            state_in_intra_q2, state_out_intra_q2, factored_next_state)

        # For next timestep
        obs = next_obs
        state_in_pi = state_out_pi
        state_in_beta = state_out_beta
        state_in_inter_q1 = state_out_inter_q1
        state_in_inter_q2 = state_out_inter_q2
        state_in_intra_q1 = state_out_intra_q1
        state_in_intra_q2 = state_out_intra_q2

        # End of trajectory handling
        if done or ep_len == env.max_episode_steps:
            agent.log[args.log_name].info("Train Returns: {:.3f} at iteration {}".format(ep_reward, total_step_count))
            agent.tb_writer.log_data("episodic_reward", total_step_count, ep_reward)

            obs, ep_reward, ep_len, beta = env.reset(), 0, 0, True

            if args.env_name.find("MiniGrid") != -1:
                obs = obs['image']

            state_in_pi, state_in_beta, state_in_inter_q1, state_in_inter_q2, state_in_intra_q1, state_in_intra_q2 = \
                initialize_lstm_state(args)

        # Update handling
        if (total_step_count + 1) % args.update_every == 0 and total_step_count > args.start_steps:
            for j in range(args.update_every):
                batch = replay_buffer.sample_batch(args.batch_size)
                agent.update_loss_soc(data=batch)

        if agent.iteration % args.save_model_every == 0:
            path = args.model_dir + args.exp_name + "_model_it_" + str(agent.iteration)
            torch.save(agent.state_dict(), path)
