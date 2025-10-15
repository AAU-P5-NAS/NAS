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
        self.max_actions_per_episode = (
            max_layers
            / 2  # (an action adds a layer and an activation function which itself is a layer)
        )
        self.data_importer = DataImporter(max_per_class=100)
        self.loader_tuple = self.data_importer.get_as_cnn(batch_size=64, test_split=0.2)
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        self.console = Console()
        self.current_network_config = NetworkConfig(layers=[])
        self.actions_taken = 0  # Track steps in episode
        self.sum_reward = 0.0
        self.step_count = 0

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
        self.step_count += 1

        try:
            new_architecture, error = self.get_architecture(action_logits)
            shouldEvaluate = (
                True
                if error is not None or self.actions_taken >= self.max_actions_per_episode
                else False
            )
            if shouldEvaluate:
                reward = self._evaluate_architecture(new_architecture)
                terminated = True
                truncated = False
                self.actions_taken = 0  # Reset for next episode
            else:
                reward = 0.5
                terminated = False
                truncated = False
        except Exception as e:
            self.console.print(f"[bold red]Exception occurred: {e}[/bold red]")
            reward = 0.0
            terminated = True
            truncated = False

        info = self._get_info()
        obs = self._get_observation()

        if self.step_count == 50:
            avg_reward = self.sum_reward / self.step_count
            self.console.print(
                f"[bold cyan]Running average reward over last {self.step_count} steps: {avg_reward:.4f}[/bold cyan]"
            )
            self.sum_reward = 0.0
            self.step_count = 0
        else:
            self.sum_reward += reward

        return obs, reward, terminated, truncated, info

    def get_architecture(
        self, action_logits: np.ndarray
    ) -> tuple[NetworkConfig, Optional[Exception]]:
        """Build a new architecture based on the agent's action logits.

        :Args:
        - action_logits: The logits produced by the agent's policy network.

        :Returns:
        - tuple: (new_network_config, error)

        """
        try:
            observation = self._get_observation()
            action_builder = ActionBuilder(10, "add_layer_sequential")
            action = action_builder.build_action(
                action_output=action_logits, observation=observation
            )
            new_network_config = arch_builder(
                actions=action, partial_arch=self.current_network_config
            )
            self.current_network_config = new_network_config
            return new_network_config, None
        except Exception:
            return self.current_network_config, Exception("No more layers can be added")

    def train_classifier(
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
        )
        num_epochs = 5
        start_time = time.time()

        for epoch in range(num_epochs):
            start_time_epoch = time.time()
            try:
                with self.console.status(
                    f"[bold blue]Training CNN, epoch {epoch + 1}/{num_epochs}"
                ):
                    trainer.train()

                end_time_epoch = time.time()
                epoch_time = end_time_epoch - start_time_epoch
                if epoch_time > 20:
                    self.console.print(
                        f"[bold red]Epoch {epoch + 1} took too long ({epoch_time:.1f}s) - likely hung[/bold red]"
                    )
                    return -3.0  # Penalty for hung training
            except Exception as e:
                self.console.print(
                    f"[bold red]Training failed at epoch {epoch + 1}: {e}[/bold red]"
                )
                return -3.0  # Penalty for failed training

        end_time = time.time()
        training_time = end_time - start_time
        if training_time > 60:
            self.console.print(
                f"[bold red]Training took too long ({training_time:.1f}s) - likely hung[/bold red]"
            )
            return -3.0  # Penalty for hung training
        elif training_time < 0.005:
            self.console.print("[bold yellow]Training too fast - likely failed[/bold yellow]")
            return -2.0  # Penalty for failed training

        self.console.log(
            f"[bold green]Training completed in {training_time:.2f} seconds[/bold green]"
        )

        with self.console.status("[bold blue]Evaluating CNN on test set..."):
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
        if len(new_architecture.layers) == 0:
            self.console.print("[bold red]No layers in architecture - giving penalty[/bold red]")
            return -5.0

        try:
            cnn_builder = CNNBuilder(rl_config=new_architecture)
            architecture = cnn_builder.build()
            optimizer = torch.optim.SGD(architecture.parameters(), lr=0.001, momentum=0.9)

            training_results = self.train_classifier(
                dataloaders=self.loader_tuple,
                model=architecture,
                loss_function=CrossEntropyLoss(),
                optimizer=optimizer,
            )
            if isinstance(training_results, Metrics):
                reward = self._calculate_reward(training_results)
                return reward
            else:
                return training_results  # Already a penalty value

        except Exception as e:
            self.console.print(f"[bold red]Architecture evaluation failed: {e}[/bold red]")
            return -5.0  # Penalty for invalid architecture

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
        self.console.print(f"[bold magenta]→ Reward: {reward:.4f}[/bold magenta]")

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
