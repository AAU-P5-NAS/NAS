import gymnasium as gym
import numpy as np
from gymnasium import spaces
from enum import Enum


class Operations(Enum):
    NO_OP = 0
    ADD_LAYER = 1
    REMOVE_LAYER = 2
    INCREASE_NEURONS = 3
    DECREASE_NEURONS = 4
    CHANGE_ACTIVATION_FUNCTION = 5


class CustomEnv(gym.Env):
    """A custom environment for reinforcement learning."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, render_mode="console", max_layers=10, max_neurons=1024):
        super().__init__()

        self.render_mode = render_mode

        N_ACTIONS = Operations.__len__()
        self.action_space = spaces.MultiDiscrete([N_ACTIONS, max_layers, max_neurons])

        self.observation_space = spaces.Dict(
            {
                "layers": spaces.Discrete(max_layers),
                "neurons": spaces.Discrete(max_neurons),
                "accuracy": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

    def render(self):
        print("haha!")

    def close(self):
        if self.render_mode == "console":
            pass
