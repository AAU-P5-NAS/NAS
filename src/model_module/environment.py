from __future__ import annotations
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any, Dict, List, Tuple, Optional
import torch
from src.model_module.action_builder import ActionBuilder
from src.classification_module.metrics import Metrics
from src.classification_module.reward import RewardCalculator
from src.classification_module.train import Trainer
from src.data_module.importer import DataImporter
from src.utils.cnn_builder import CNNBuilder, flatten_cnn_config
from src.utils.network_utils import (
    LayerType,
    NetworkConfig,
    OutChannels,
    KernelSize,
    Stride,
    LinearUnits,
    PoolMode,
    ActivationFunction,
)
from src.utils.arch_builder import arch_builder
from torch.nn import CrossEntropyLoss
from rich.console import Console

standard_actions = {
    "REMOVE_LAYER",
    "MODIFY_LAYER",
    "ADD_LAYER",
    "DO_NOTHING",
}


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
    render_mode: str
    max_layers: int

    def __init__(self, render_mode: str = "console", max_layers: int = 10):
        super().__init__()

        self.render_mode = render_mode
        self.max_layers = max_layers
        self.data_importer = DataImporter(max_per_class=30)
        self.loader_tuple = self.data_importer.get_as_cnn(batch_size=64, test_split=0.2)
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        self.console = Console()
        self.current_network_config = NetworkConfig(layers=[])

    def _get_action_space(self) -> spaces.Space:
        output_actions = (
            len(standard_actions)
            + len(LayerType)
            + (self.max_layers - 1)  # One for each index (which index to apply action on)
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
        # Add +1 to each enum length to include an "empty/unused" value
        observation_space_vector.append(len(LayerType))
        observation_space_vector.append(len(OutChannels))
        observation_space_vector.append(len(KernelSize))
        observation_space_vector.append(len(Stride))
        observation_space_vector.append(len(PoolMode))
        observation_space_vector.append(len(ActivationFunction))
        observation_space_vector.append(len(LinearUnits))
        observation_space_vector *= self.max_layers

        return spaces.MultiDiscrete(observation_space_vector)

    def _get_observation(self) -> List[int]:
        """Retrieve state from other agent.
        Returns:
        """
        flattened_obs = flatten_cnn_config(self.current_network_config, self.max_layers)
        return flattened_obs

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

    def step(self, action_logits: np.ndarray) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Execute one timestep within the environment.

        Args:
            action: The action to take and the index of the layer to apply the action on.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # update the current observation
        action_builder = ActionBuilder(10, "add_layer_sequential")
        obs = self._get_observation()
        action = action_builder.build_action(action_output=action_logits, observation=obs)
        network_configuration = arch_builder(
            actions=action, partial_arch=self.current_network_config
        )
        self.current_network_config = network_configuration

        cnn_builder = CNNBuilder(rl_config=network_configuration)
        architecture = cnn_builder.build()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        architecture.to(device)
        optimizer = torch.optim.SGD(architecture.parameters(), lr=0.01, momentum=0.9)
        trainer = Trainer(
            dataloaders=self.loader_tuple,
            model=architecture,
            loss_function=CrossEntropyLoss(),
            optimizer=optimizer,
        )

        num_epochs = 5
        start_time = time.time()
        for epoch in range(num_epochs):
            with self.console.status(f"[bold blue]Training CNN, epoch {epoch + 1}/{num_epochs}"):
                trainer.train()
        end_time = time.time()
        training_time = end_time - start_time
        self.console.log(
            f"[bold green]Training completed in {training_time:.2f} seconds[/bold green]"
        )
        with self.console.status("[bold blue]Evaluating CNN on test set..."):
            metrics: Metrics = trainer.test()
        # self.console.log(f"[bold green]Evaluation completed. Metrics: {metrics}[/bold green]")
        metrics.training_time = training_time
        # self.console.print(f"[bold yellow]Test Metrics:[/bold yellow] {metrics.model_dump()}")

        terminated = self._should_terminate()
        truncated = self._has_truncated()

        reward = self._calculate_reward(metrics)
        info = self._get_info()

        obs = self._get_observation()
        return obs, reward, terminated, truncated, info

    def _should_terminate(self) -> bool:
        return False
        raise NotImplementedError

    def _has_truncated(self) -> bool:
        return False
        raise NotImplementedError

    def _calculate_reward(self, metrics: Metrics) -> float:
        rewardCalculator = RewardCalculator()
        reward: float = rewardCalculator.compute_reward(metrics)
        self.console.print(f"[bold magenta]Reward:[/bold magenta] {reward}")
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
