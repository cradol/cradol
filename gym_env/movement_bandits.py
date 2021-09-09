import logging
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math

logger = logging.getLogger(__name__)


class MovementBandits(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, add_action_in_obs=False):
        # new action space = [left, right]
        self.action_space = spaces.Discrete(5)
        self.random_data_len = 10
        if add_action_in_obs:
            self.observation_space = spaces.Box(-10000000, 10000000, shape=(7 + self.random_data_len,))
        else:
            self.observation_space = spaces.Box(-10000000, 10000000, shape=(7 + self.random_data_len,))
        self.add_action_in_obs = add_action_in_obs

        self.realgoal = 0
        self.radius = 175
        self.max_episode_steps = 50
        self.iteration = 0
        self.seed(1)
        self.viewer = None
        self.steps_beyond_done = None

        self.reset()

        # Just need to initialize the relevant attributes
        self._configure()

    def draw_and_set_task(self, constraint=None, seed=None):
        """
        Draw a new task and set the environment to that task.

        Args:
            seed: Random seed that is used to generate task
            contraint:  Should be e.g. "B-Y"
        """
        if seed is None:
            seed = self.np_random.randint(9223372036854775807)

        _rnd, seed1 = seeding.np_random(seed)
        if constraint is None:
            constraint = _rnd.choice([0, 1])
        # constraint should be either 0 or 1
        self.realgoal = int(constraint)

        return seed1

    def _configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.goals = []
        self.state = [200.0, 200.0]
        self.random_data = []

        for x in range(2):
            theta = self.np_random.uniform(0, 360)
            theta = theta * math.pi / 180.0
            x = self.radius * math.cos(theta) + self.state[0]
            y = self.radius * math.sin(theta) + self.state[1]
            self.goals.append((x, y))

        for i in range(self.random_data_len):
            self.random_data.append(np.random.uniform())

        return self.obs(0)

    def step(self, action):
        done = False
        if action == 1:
            self.state[0] += 20
        if action == 2:
            self.state[0] -= 20
        if action == 3:
            self.state[1] += 20
        if action == 4:
            self.state[1] -= 20

        distance = np.mean(abs(self.state[0] - self.goals[self.realgoal][0]) ** 2 + abs(
            self.state[1] - self.goals[self.realgoal][1]) ** 2)

        if distance < 2500:
            reward = 1
            done = True
        else:
            reward = 0

        return self.obs(action), reward, done, {}

    def obs(self, action):
        obs = np.reshape(np.array([self.state] + self.goals), (-1,)) / 400
        obs = np.concatenate([obs, [self.realgoal], self.random_data])

        if self.add_action_in_obs:
            obs = np.concatenate([obs, [action]])
        return obs

    def render(self, mode='rgb_array', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 400
        screen_height = 400

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height, display=self.display)
            self.man_trans = rendering.Transform()
            self.man = rendering.make_circle(10)
            self.man.add_attr(self.man_trans)
            self.man.set_color(.5, .8, .5)
            self.viewer.add_geom(self.man)

            self.goal_trans = []
            for g in range(len(self.goals)):
                self.goal_trans.append(rendering.Transform())
                self.goal = rendering.make_circle(20)
                self.goal.add_attr(self.goal_trans[g])
                self.viewer.add_geom(self.goal)
                self.goal.set_color(.5, .5, g * 0.8)

        self.man_trans.set_translation(self.state[0], self.state[1])
        for g in range(len(self.goals)):
            self.goal_trans[g].set_translation(self.goals[g][0], self.goals[g][1])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
