from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any, Dict, Tuple, Optional, List
from pydantic import BaseModel as type_check
from utils.CNNBuilder import (
    LayerType,
    OutChannels,
    KernelSize,
    Stride,
    LinearUnits,
    PoolMode,
    ActivationFunction,
)

standard_actions = {
    "REMOVE_LAYER",
    "MODIFY_LAYER",
    "ADD_LAYER",
    "DO_NOTHING",
}


class CustomEnv(gym.Env, type_check):
    """
    🎯 What skill should the agent learn?
        Training other agents.

    👀 What information does the agent need?
        The current state of the other agent and its performance.

    🎮 What actions can the agent take?
        Actions to chane the other agents architecture (see enum class Operations).

    🏆 How do we measure success?
        By maximizing the other agents performance.

    ⏰ When should episodes end?
        When the other agent has been trained a given amount of times. Or when improvement over a given amount of steps shows little to no improvement.
    """

    metadata = {"render.modes": ["human"]}
    render_mode: str
    max_layers: int

    def __init__(self, render_mode: str = "console", max_layers: int = 10):
        super().__init__()

        self.render_mode = render_mode
        self.max_layers = max_layers

        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

    def _get_action_space(self) -> spaces.Space:
        output_actions = (
            len(standard_actions)
            + len(LayerType)
            + (self.max_layers - 1)  # One for each index
            + len(OutChannels)
            + len(KernelSize)
            + len(Stride)
            + len(LinearUnits)
            + len(PoolMode)
            + len(ActivationFunction)
        )

        return spaces.Box(low=0, high=1, shape=(output_actions,), dtype=np.float32)

    def _get_observation_space(self) -> spaces.Space:
        observation_space_vector: List[int] = []
        observation_space_vector.append(len(LayerType))
        observation_space_vector.append(len(OutChannels))
        observation_space_vector.append(len(KernelSize))
        observation_space_vector.append(len(Stride))
        observation_space_vector.append(len(PoolMode))
        observation_space_vector.append(len(ActivationFunction))
        observation_space_vector.append(len(LinearUnits))
        observation_space_vector *= self.max_layers

        return spaces.MultiDiscrete(observation_space_vector)

    def _get_observation(self):
        """Retrieve state from other agent.

        Returns:

        """
        return []

    def _get_info(self) -> Dict[str, Any]:
        """Compute auxiliary information for debugging.

        Returns:

        """
        return {}

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (currently unused)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: List[float]) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Execute one timestep within the environment.

        Args:
            action: The action to take and the index of the layer to apply the action on.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """

        terminated = self._should_terminate()
        truncated = self._has_truncated()

        reward = self._calculate_reward()

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _should_terminate(self) -> bool:
        raise NotImplementedError

    def _has_truncated(self) -> bool:
        raise NotImplementedError

    def _calculate_reward(self) -> float:
        reward: float = 0
        return reward

    def render(self):
        print(self._get_observation())

    def close(self):
        if self.render_mode == "console":
            pass
        else:
            raise NotImplementedError


""" 
    🎯 What skill should the agent learn?
    Navigate through a maze?
    Balance and control a system?
    Optimize resource allocation?
    Play a strategic game?

    👀 What information does the agent need?
    Position and velocity?
    Current state of the system?
    Historical data?
    Partial or full observability?

    🎮 What actions can the agent take?
    Discrete choices (move up/down/left/right)?
    Continuous control (steering angle, throttle)?
    Multiple simultaneous actions?

    🏆 How do we measure success?
    Reaching a specific goal?
    Minimizing time or energy?
    Maximizing a score?
    Avoiding failures?

    ⏰ When should episodes end?
    Task completion (success/failure)?
    Time limits?
    Safety constraints?
"""
