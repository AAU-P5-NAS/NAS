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
from torch.nn import CrossEntropyLoss, Module
from rich.console import Console
from stable_baselines3.common.logger import Logger
from torch.utils.tensorboard import SummaryWriter

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
    logdir: str
    info: Dict[str, Any] = {}
    actions_taken: int = 0  # Track steps in episode
    sum_reward: float = 0.0
    sum_accuracy: float = 0.0
    evaluation_count: int = 0
    step_count: int = 0

    newest_reward: Optional[float] = None
    newest_metrics: Optional[Metrics] = None
    newest_architecture: Optional[Module] = None
    newest_actions_taken_on_evaluation: Optional[int] = None

    evaluated_this_step: bool = False

    def __init__(self, logdir: str, render_mode: str = "console", max_layers: int = 10):
        super().__init__()

        self.render_mode = render_mode
        self.max_layers = max_layers
        self.max_actions_per_episode = (
            max_layers
            / 2  # (an action adds a layer and an activation function which itself is a layer)
        )
        self.data_importer = DataImporter(max_per_class=1000)
        self.loader_tuple = self.data_importer.get_as_cnn(batch_size=64, test_split=0.2)
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        self.console = Console()
        self.current_network_config = NetworkConfig(layers=[])
        self.actions_taken = 0  # Track steps in episode
        self.sum_reward = 0.0
        self.sum_accuracy = 0.0
        self.evaluation_count = 0
        self.logdir = logdir

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

    def _write_summary(self, logger: Logger, summary_writer: SummaryWriter) -> None:
        """
        Write evaluation metrics and reward to TensorBoard summary, if evaluated_this_step is True.
        """

        if self.evaluated_this_step is False:
            return

        def record_optional(name: str, metric: Optional[float]) -> None:
            if metric is not None:
                logger.record(name, metric)

        record_optional("Custom/Reward", self.newest_reward)
        record_optional("Custom/Actions Taken", self.newest_actions_taken_on_evaluation)

        if self.newest_metrics is not None:
            record_optional("Custom/Test Loss", self.newest_metrics.test_loss)
            record_optional("Custom/Accuracy", self.newest_metrics.accuracy)
            record_optional("Custom/Precision", self.newest_metrics.precision)
            record_optional("Custom/Recall", self.newest_metrics.recall)
            record_optional("Custom/F1 Score", self.newest_metrics.f1_score)
            record_optional("Custom/FLOPs", self.newest_metrics.flops)
            record_optional("Custom/Runtime", self.newest_metrics.runtime)
            record_optional("Custom/Architecture Size", self.newest_metrics.architecture_size)

        logger.dump(step=self.step_count)

        if self.newest_architecture is not None:
            summary_writer.add_graph(self.newest_architecture, torch.zeros(1, 1, 28, 28))

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
        return observation, self.info

    def step(self, action_logits: np.ndarray) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Execute one timestep within the environment.

        :Args:
        - action: The logits produced by the agent's policy network.

        :Returns:
        - tuple: (observation, reward, terminated, truncated, info)
        """

        self.evaluated_this_step = False

        self.info = {}
        self.step_count += 1
        self.actions_taken += 1

        with self.console.status(
            f"[bold blue]Training model on action number {self.evaluation_count + 1} ...[/bold blue]\nCurrent sum reward: {self.sum_reward}"
        ):
            reward = 0.0
            terminated = False
            truncated = False

            try:
                new_architecture, error = self.get_architecture(action_logits)
                shouldEvaluate = (
                    True
                    if error is not None or self.actions_taken >= self.max_actions_per_episode
                    else False
                )
                if shouldEvaluate:
                    self.evaluated_this_step = True

                    reward = self._evaluate_architecture(new_architecture)
                    terminated = True
                    self.newest_actions_taken_on_evaluation = self.actions_taken
                    self.actions_taken = 0  # Reset for next episode
            except Exception as e:
                self.console.print(f"[bold red]Exception occurred: {e}[/bold red]")

        obs = self._get_observation()

        return obs, reward, terminated, truncated, self.info

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
        num_epochs = 15
        start_time = time.time()

        for epoch in range(num_epochs):
            with self.console.status(
                f"[bold blue]Training epoch {epoch + 1}/{num_epochs}...[/bold blue]\n"
            ):
                try:
                    trainer.train()

                except Exception as e:
                    self.console.print(
                        f"[bold red]Training failed at epoch {epoch + 1}: {e}[/bold red]"
                    )
                    return -3.0  # Penalty for failed training

        end_time = time.time()
        training_time = end_time - start_time
        metrics = trainer.test()

        metrics.runtime = training_time
        metrics.training_time = training_time

        self.newest_metrics = metrics

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
            self.newest_architecture = cnn_builder.build()
            optimizer = torch.optim.SGD(
                self.newest_architecture.parameters(), lr=0.001, momentum=0.9
            )

            training_results = self.train_classifier(
                dataloaders=self.loader_tuple,
                model=self.newest_architecture,
                loss_function=CrossEntropyLoss(),
                optimizer=optimizer,
            )
            if isinstance(training_results, Metrics):
                self.console.print(
                    f"[bold green]Accuracy: {training_results.accuracy}[/bold green]"
                )
                reward = self._calculate_reward(training_results)

                self.console.print(f"[bold green]Reward: {reward}[/bold green]")
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

        self.newest_reward = float(reward)
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
