import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CustomEnv(gym.Env):
    """A custom environment for reinforcement learning."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, render_mode="console", max_layers=10, max_neurons=1024):
        super().__init__()

        self.render_mode = render_mode

        N_ACTIONS = 5
        self.action_space = spaces.Discrete(N_ACTIONS)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
