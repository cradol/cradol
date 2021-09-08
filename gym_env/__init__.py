import gym
from gym.envs.registration import register
from gym_env.movement_bandits import MovementBandits


def make_envs(env_name, args):
    if env_name == "MovementBandits-v0":
        env = MovementBandits()
    elif env_name == "Reacher-v2":
        from gym_env.reacher import ReacherEnv
        env = ReacherEnv()
        env.max_episode_steps = args.max_episode_steps
    elif env_name.find("MiniGrid") != -1:
        import gym_minigrid
        env = gym.make(env_name)
        env.max_episode_steps = env.max_steps
    else:
        env = gym.make(env_name)
        env.max_episode_steps = args.max_episode_steps
    return env


register(
    id='MovementBandits-v0',
    entry_point='gym_env.movement_bandits:MovementBandits',
    max_episode_steps=50)
