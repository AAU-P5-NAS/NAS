import gymnasium as gym
import numpy as np
from gymnasium import spaces
from enum import Enum
from typing import Any, Dict, Tuple, Optional
from abc import ABC, abstractmethod

from utils import CNNBuilder

N_LAYER_TYPES = CNNBuilder.LayerType.__len__()
N_LINEAR_UNITS = CNNBuilder.LinearUnits.__len__()
N_OUT_CHANNELS = CNNBuilder.OutChannels.__len__()
N_KERNAL_SIZES = CNNBuilder.KernelSize.__len__()
N_STRIDES = CNNBuilder.Stride.__len__()
N_POOL_MODES = CNNBuilder.PoolMode.__len__()
N_ACTIVATION_FUNCTIONS = CNNBuilder.ActivationFunction.__len__()
IS_PRESENT = 2

LAYER_FIELDS = [
    N_LAYER_TYPES,
    N_LINEAR_UNITS,
    N_OUT_CHANNELS,
    N_KERNAL_SIZES,
    N_STRIDES,
    N_POOL_MODES,
    N_ACTIVATION_FUNCTIONS,
    IS_PRESENT,  # Binary value
]


# CONSIDER MAKING THIS A FLAG ENUM TO ALLOW THE MODEL TO DO MULTIPLE ACTIONS AT ONCE ?
class Operation(Enum):
    NO_OP = 0
    ADD_LAYER = 1
    REMOVE_LAYER = 2
    CHANGE_LAYER_TYPE = 3
    CHANGE_LINEAR_UNITS = 4
    CHANGE_OUT_CHANNELS = 5
    CHANGE_KERNAL_SIZE = 6
    CHANGE_STRIDE = 7
    CHANGE_POOL_MODE = 8
    CHANGE_ACTIVATION_FUNCTION = (9,)
    CHANGE_LEARNING_RATE = (10,)


N_OPERATIONS = Operation.__len__()


class Action:
    operation: Operation
    layer_index: int

    def __init__(self, operation: Operation, layer_index: int):
        self.operation = operation
        self.layer_index = layer_index


class AgentABC(ABC):
    class StepReturn:
        class Performance:
            accuracy: float
            training_epochs: int

        took_illegal_action: bool
        performance: Optional[Performance]

    @abstractmethod
    def on_step_from_parent_environment(self, action: Action) -> StepReturn:
        pass

    @abstractmethod
    def get_observation_for_parent_environment(self) -> Any:
        pass

    @abstractmethod
    def get_info_for_parent_environment(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def on_reset_from_parent_environment(self):
        pass


class CustomEnv(gym.Env):
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

    def __init__(self, other_agent: AgentABC, render_mode: str = "console", max_layers: int = 10):
        super().__init__()

        self.check_validation()

        self.render_mode = render_mode
        self.max_layers = max_layers
        self.other_agent = other_agent

        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

    def _get_action_space(self) -> spaces.Space:
        return spaces.MultiDiscrete([N_OPERATIONS, self.max_layers])

    def _get_observation_space(self) -> spaces.Space:
        nvec = []
        for _ in range(self.max_layers):
            for layer_field in LAYER_FIELDS:
                nvec.append(layer_field)

        nvec = np.array(nvec, dtype=np.int64)

        return spaces.Dict(
            {
                "layers": spaces.MultiDiscrete(nvec),
                "accuracy": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

    def _get_observation(self):
        """Retrieve state from other agent.

        Returns:

        """
        return self.other_agent.get_observation_for_parent_environment()

    def _get_info(self) -> Dict[str, Any]:
        """Compute auxiliary information for debugging.

        Returns:

        """
        return self.other_agent.get_info_for_parent_environment()

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

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Execute one timestep within the environment.

        Args:
            action: The action to take and the index of the layer to apply the action on.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """

        operation: Operation = Operation(action[0])
        layer_index: int = action[1]

        stepReturn: AgentABC.StepReturn = self.other_agent.on_step_from_parent_environment(action)

        terminated = self._should_terminate()
        truncated = self._has_truncated()

        reward = self._calculate_reward(stepReturn)

        observation = self.other_agent.get_observation_for_parent_environment()
        info = self.other_agent.get_info_for_parent_environment()

        return observation, reward, terminated, truncated, info

    def _should_terminate(self) -> bool:
        raise NotImplementedError

    def _has_truncated(self) -> bool:
        raise NotImplementedError

    def _calculate_reward(self, other_agent_step_return: AgentABC.StepReturn) -> float:
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        if self.render_mode == "console":
            pass
        else:
            raise NotImplementedError

    def check_validation(self):
        for layer_field in LAYER_FIELDS:
            if layer_field <= 0:
                raise ValueError(
                    "Each entry in LAYER_FIELDS must be an integer >= 1 (a cardinality)."
                )


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
