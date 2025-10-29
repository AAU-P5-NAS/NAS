from __future__ import annotations
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any, Dict, List, Tuple, Optional
import torch
from model_module.action_builder import transform_logits_to_action
from src.classification_module.metrics import Metrics
from src.classification_module.reward import RewardCalculator
from src.classification_module.train import Trainer
from src.data_module.importer import DataImporter, DatasetOption
from src.utils.cnn_builder import CNNBuilder, flatten_cnn_config
from src.utils.network_utils import (
    LayerType,
    NetworkConfig,
    OutChannels,
    KernelSize,
    StandardAction,
    Stride,
    LinearUnits,
    PoolMode,
    ActivationFunction,
)
from torch.nn import CrossEntropyLoss
from rich.console import Console

from utils.action_builder_utils import get_logit_slices


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

    def __init__(self, render_mode: str = "console", max_layers: int = 16):
        super().__init__()

        self.render_mode = render_mode
        self.max_layers = max_layers
        self.max_actions_per_episode = (
            max_layers
            / 2  # (an action adds a layer and an activation function which itself is a layer)
        )
        self.data_importer = DataImporter(dataset_option=DatasetOption.EMNIST_BALANCED)
        self.logit_slices = get_logit_slices(self.max_layers)
        self.loader_tuple = self.data_importer.get_dataloaders(batch_size=64)
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        self.console = Console()
        self.current_network_config = NetworkConfig(layers=[])
        self.actions_taken = 0  # Track steps in episode
        self.sum_reward = 0.0
        self.sum_accuracy = 0.0
        self.evaluation_count = 0

    def _get_action_space(self) -> spaces.Space:
        output_actions = (
            len(StandardAction)
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
        observation_space_vector.append(len(LayerType))
        observation_space_vector.append(len(OutChannels))
        observation_space_vector.append(len(KernelSize))
        observation_space_vector.append(len(Stride))
        observation_space_vector.append(len(PoolMode))
        observation_space_vector.append(len(ActivationFunction))
        observation_space_vector.append(len(LinearUnits))
        observation_space_vector *= self.max_layers

        return spaces.MultiDiscrete(observation_space_vector)

    def _get_observation(self) -> np.ndarray:
        flattened_obs = flatten_cnn_config(self.current_network_config, self.max_layers)
        return np.array(flattened_obs, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Compute auxiliary information for debugging.

        Returns:

        """
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Start a new episode.

        :Args:
        - seed: Random seed for reproducible episodes
        - options: Additional configuration (currently unused)

        :Returns:
        - tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Reset episode state
        self.actions_taken = 0
        self.current_network_config = NetworkConfig(layers=[])
        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action_logits: np.ndarray) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Execute one timestep within the environment.

        :Args:
        - action: The logits produced by the agent's policy network.

        :Returns:
        - tuple: (observation, reward, terminated, truncated, info)
        """
        self.actions_taken += 1

        new_architecture, should_evaluate = self._get_new_architecture(action_logits)

        if should_evaluate:
            with self.console.status(
                f"[bold blue]Training model on action number {self.evaluation_count} ...[/bold blue]"
            ):
                reward = self._evaluate_architecture(new_architecture)
                terminated = True
                truncated = False
                self.actions_taken = 0  # Reset for next episode
        else:
            reward = 0.05
            terminated = False
            truncated = False

        info = self._get_info()
        obs = self._get_observation()

        return obs, reward, terminated, truncated, info

    def _get_new_architecture(self, action_logits: np.ndarray):
        """Build a new architecture based on the agent's action logits.

        :Args:
        - action_logits: The logits produced by the agent's policy network.

        :Returns:
        - tuple: (new_network_config, should_evaluate)
        """
        observation = self._get_observation()
        action_to_apply = transform_logits_to_action(action_logits, observation, self.max_layers)

        if action_to_apply is None:
            return self.current_network_config, True  # Stop and evaluate

        new_network_config = self.current_network_config.extend(
            action=action_to_apply.to_int_list(), partial_arch=self.current_network_config
        )
        self.current_network_config = new_network_config
        return new_network_config, False  # Do not evaluate yet

    def _train_classifier(
        self,
        dataloaders: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader],
        model: torch.nn.Module,
        loss_function: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
    ):
        """Train and evaluate the given model, returning the evaluation metrics. Expects that all inputs are not already moved to device.

        :Args:
        - dataloaders: Tuple of (train_dataloader, test_dataloader).
        - model: The CNN model to train.
        - loss_function: The loss function to use (e.g., CrossEntropyLoss).
        - optimizer: The optimizer to use (e.g., SGD, Adam).

        :Returns:
        - Metrics | float: The evaluation metrics after training, or a penalty float if training failed.
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer = Trainer(
            dataloaders=dataloaders,
            model=model.to(device),
            loss_function=loss_function.to(device),
            optimizer=optimizer,
            num_classes=self.data_importer.get_num_classes()[0],
        )
        num_epochs = 10
        start_time = time.time()

        for epoch in range(num_epochs):
            self.console.status(f"[bold blue]Epoch {epoch + 1}/{num_epochs}[/bold blue]")
            trainer.train()

        end_time = time.time()
        training_time = end_time - start_time
        metrics = trainer.test()

        metrics.runtime = training_time
        metrics.training_time = training_time

        return metrics

    def _evaluate_architecture(self, new_architecture: NetworkConfig) -> float:
        """Evaluate the given architecture by training and testing it, returning the computed reward.

        :Args:
        - new_architecture: The CNN architecture to evaluate.

        :Returns:
        - float: The computed reward based on evaluation metrics.
        """
        cnn_builder = CNNBuilder(
            rl_config=new_architecture, num_classes=self.data_importer.get_num_classes()[0]
        )
        architecture = cnn_builder.build()
        optimizer = torch.optim.SGD(architecture.parameters(), lr=0.001, momentum=0.9)
        training_results = self._train_classifier(
            dataloaders=self.loader_tuple,
            model=architecture,
            loss_function=CrossEntropyLoss(),
            optimizer=optimizer,
        )

        if isinstance(training_results, Metrics):
            self.console.print(f"[bold green]Accuracy: {training_results.accuracy}[/bold green]")
            reward = self._calculate_reward(training_results)
            self.console.print(f"[bold green]Reward: {reward}[/bold green]")
            self.console.print("[bold green]Architecture:[/bold green]")
            for i, layer in enumerate(new_architecture.layers, start=1):
                self.console.print(f"  [bold yellow]Layer {i}:[/bold yellow] {layer}")
            return reward
        else:
            return training_results  # Already a penalty value

    def _should_terminate(self) -> bool:
        # Termination is now handled in step() method
        return False

    def _has_truncated(self) -> bool:
        # Truncation is now handled in step() method
        return False

    def _calculate_reward(self, metrics: Metrics) -> float:
        """Calculate the reward based on evaluation metrics.

        :Args:
        - metrics: The evaluation metrics from testing the architecture.

        :Returns:
        - float: The computed reward.
        """
        rewardCalculator = RewardCalculator()
        reward: float = rewardCalculator.compute_reward(metrics)
        self.evaluation_count += 1

        if self.evaluation_count >= 50:
            avg_reward = self.sum_reward / self.evaluation_count
            avg_accuracy = self.sum_accuracy / self.evaluation_count
            self.console.print(
                f"[bold cyan]Average reward over last {self.evaluation_count} actions: {avg_reward:.4f}[/bold cyan]"
            )
            self.console.print(
                f"[bold cyan]Average accuracy over last {self.evaluation_count} actions: {avg_accuracy:.4f}[/bold cyan]"
            )
            self.sum_reward = 0.0
            self.sum_accuracy = 0.0
            self.evaluation_count = 0
        else:
            self.sum_reward += reward
            if hasattr(metrics, "accuracy") and metrics.accuracy is not None:
                self.sum_accuracy += metrics.accuracy
            else:
                self.sum_accuracy += 0.0

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
