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
        self.max_steps_per_episode = 5  # Agent can take up to 5 actions per architecture
        self.data_importer = DataImporter(max_per_class=50)
        self.loader_tuple = self.data_importer.get_as_cnn(batch_size=64, test_split=0.2)
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        self.console = Console()
        self.current_network_config = NetworkConfig(layers=[])
        self.step_count = 0  # Track steps in episode

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

    def _get_observation(self) -> np.ndarray:
        flattened_obs = flatten_cnn_config(self.current_network_config, self.max_layers)
        return np.array(flattened_obs, dtype=np.float32)

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

        # Reset episode state
        self.step_count = 0
        self.current_network_config = NetworkConfig(layers=[])

        observation = self._get_observation()
        assert isinstance(observation, np.ndarray), f"obs must be numpy, got {type(observation)}"
        info = self._get_info()

        return observation, info

    def step(self, action_logits: np.ndarray) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Execute one timestep within the environment.

        Args:
            action: The action to take and the index of the layer to apply the action on.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self.step_count += 1
        """ self.console.print(
            f"[bold cyan]Step {self.step_count}/{self.max_steps_per_episode}[/bold cyan] - Current layers: {len(self.current_network_config.layers)}"
        ) """

        # Try to build action - if it fails, agent is done building
        action_builder = ActionBuilder(10, "add_layer_sequential")
        obs = self._get_observation()
        assert isinstance(obs, np.ndarray), f"obs must be numpy, got {type(obs)}"
        try:
            action = action_builder.build_action(action_output=action_logits, observation=obs)
            # If action was successful, update the architecture
            network_configuration = arch_builder(
                actions=action, partial_arch=self.current_network_config
            )
            self.current_network_config = network_configuration

            # Check if we should terminate this episode and evaluate
            if self.step_count >= self.max_steps_per_episode:
                # Episode complete - evaluate architecture and give reward
                reward = self._evaluate_architecture()
                terminated = True
                truncated = False
            else:
                # Continue building - give small positive reward for valid action
                reward = 0.1
                terminated = False
                truncated = False

        except Exception:
            # Action builder returned None/failed - agent is done building
            self.console.print(
                "[bold yellow]Agent finished building architecture (no more actions)[/bold yellow]"
            )
            reward = self._evaluate_architecture()
            terminated = True
            truncated = False

        info = self._get_info()
        obs = self._get_observation()
        return obs, reward, terminated, truncated, info

    def _evaluate_architecture(self) -> float:
        """Evaluate the current architecture and return reward"""
        if len(self.current_network_config.layers) == 0:
            self.console.print("[bold red]No layers in architecture - giving penalty[/bold red]")
            return -5.0

        try:
            cnn_builder = CNNBuilder(rl_config=self.current_network_config)
            architecture = cnn_builder.build()

            # Ensure a single explicit device and move the model BEFORE creating the optimizer.
            # If the optimizer is created while params are on CPU, its state may be on CPU and cause
            # CPU/GPU device-mismatch errors during training.
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            architecture.to(device)

            # Validate architecture has trainable parameters
            total_params = sum(p.numel() for p in architecture.parameters() if p.requires_grad)
            if total_params == 0:
                self.console.print("[bold red]Architecture has no trainable parameters[/bold red]")
                return -4.0

            # Adaptive training setup based on network depth
            num_layers = len(self.current_network_config.layers)
            if num_layers <= 2:
                # Shallow networks: higher LR, SGD works fine
                lr = 0.01
                optimizer = torch.optim.SGD(architecture.parameters(), lr=lr, momentum=0.9)
            else:
                # Deep networks: lower LR, Adam optimizer for better convergence
                lr = 0.001
                optimizer = torch.optim.Adam(architecture.parameters(), lr=lr)

            trainer = Trainer(
                dataloaders=self.loader_tuple,
                model=architecture,
                loss_function=CrossEntropyLoss(),
                optimizer=optimizer,
                device=device,
            )
            print("I REACH THIS PLACE")

            # Adaptive epochs based on network complexity
            num_layers = len(self.current_network_config.layers)
            if num_layers <= 2:
                num_epochs = 10  # Shallow networks train quickly
            else:
                num_epochs = 15  # Deep networks need more epochs
            start_time = time.time()

            # Train for fixed number of epochs with timeout protection
            for epoch in range(num_epochs):
                epoch_start = time.time()
                try:
                    with self.console.status(
                        f"[bold blue]Training CNN, epoch {epoch + 1}/{num_epochs}"
                    ):
                        trainer.train()

                    # Check for hung training (epoch taking too long)
                    epoch_time = time.time() - epoch_start
                    if epoch_time > 15:  # 15 seconds per epoch max
                        self.console.print(
                            f"[bold yellow]Epoch {epoch + 1} took {epoch_time:.1f}s - stopping early[/bold yellow]"
                        )
                        break

                except Exception as e:
                    self.console.print(
                        f"[bold red]Training failed at epoch {epoch + 1}: {e}[/bold red]"
                    )
                    return -3.0  # Penalty for training failure

            end_time = time.time()
            training_time = end_time - start_time  # Check training time validity
            if training_time > 60:  # Over 1 minute suggests hanging
                self.console.print(
                    f"[bold red]Training took too long ({training_time:.1f}s) - likely hung[/bold red]"
                )
                return -3.0  # Penalty for hung training
            elif training_time < 0.005:  # Less than 300ms suggests failed training
                self.console.print("[bold yellow]Training too fast - likely failed[/bold yellow]")
                return -2.0  # Penalty for failed training

            self.console.log(
                f"[bold green]Training completed in {training_time:.2f} seconds[/bold green]"
            )

            with self.console.status("[bold blue]Evaluating CNN on test set..."):
                metrics: Metrics = trainer.test()

            # Set both runtime fields to ensure compatibility
            metrics.runtime = training_time
            metrics.training_time = training_time

            reward = self._calculate_reward(metrics)
            return reward

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
        rewardCalculator = RewardCalculator()
        reward: float = rewardCalculator.compute_reward(metrics)

        # Debug logging
        self.console.print(
            f"[bold magenta]Metrics - Accuracy: {getattr(metrics, 'accuracy', 'N/A'):.4f}, "
            f"F1: {getattr(metrics, 'f1_score', 'N/A'):.4f}, "
            f"Runtime: {getattr(metrics, 'runtime', 'N/A'):.2f}s, "
            f"Layers: {len(self.current_network_config.layers)}[/bold magenta]"
        )
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
