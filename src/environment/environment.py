from __future__ import annotations
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any, Dict, List, Tuple, Optional
import torch
from src.environment.metrics import Evaluator
from src.utils.hyperparameters import SLHyperParameters
from src.utils.logger import NoOpLogger, TensorboardLogger
from src.environment.reward import RewardStrategy
from src.environment.train import Trainer
from src.utils.data_importer.importer import DataImporter, DatasetOption
from src.utils.architecture import Architecture, flatten_cnn_config
from src.utils.network_config import (
    NetworkConfig,
)
from src.utils.layer_config import (
    LayerConfig,
    LayerType,
    StandardAction,
    OutChannels,
    KernelSize,
    Stride,
    LinearUnits,
    PoolMode,
    ActivationFunction,
)
from torch.nn import CrossEntropyLoss
from rich.console import Console

from src.agent.action_masking.action_masking_utils import (
    get_logit_slices,
    transform_action_indices_to_decisions,
)

MAX_LAYERS = 12


class CustomEnv(gym.Env):
    """Custom Environment that follows gymnasium interface."""

    metadata = {"render.modes": ["human"]}
    render_mode: str
    info: Dict[str, Any] = {}
    actions_taken: int = 0  # Track steps in episode
    device: str
    tb_logger: TensorboardLogger | NoOpLogger

    def __init__(
        self,
        device: str,
        hyperparameters: SLHyperParameters,
        reward_strategy: RewardStrategy,
        tb_logger: TensorboardLogger | NoOpLogger,
        dataset: DatasetOption = DatasetOption.CIFAR_10,
        render_mode: str = "console",
    ):
        super().__init__()

        self.device = device
        self.render_mode = render_mode
        self.hyperparameters = hyperparameters
        self.reward_strategy = reward_strategy
        self.tb_logger = tb_logger
        self.data_importer = DataImporter(dataset_option=dataset)

        self.logit_slices = get_logit_slices(max_layers=MAX_LAYERS)
        self.loader_tuple = self.data_importer.get_dataloaders(
            batch_size=self.hyperparameters.batch_size
        )
        self.dimensions = self.data_importer.get_dimensions()
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        self.console = Console()
        self.current_network_config = NetworkConfig(layers=[])
        self.actions_taken = 0  # Track steps in episode
        self.trainer = Trainer(
            dataloaders=self.loader_tuple,
            loss_function=CrossEntropyLoss(),
        )
        self.evaluator = Evaluator(
            num_classes=self.data_importer.get_num_classes()[0],
            dataloaders=self.loader_tuple,
            dimensions=self.dimensions,
            device=torch.device(self.device),
            loss_function=CrossEntropyLoss(),
        )

    def _get_action_space(self) -> spaces.Space:
        return spaces.MultiDiscrete(
            [
                len(StandardAction),
                len(LayerType),
                len(OutChannels),
                len(KernelSize),
                len(Stride),
                len(LinearUnits),
                len(PoolMode),
                len(ActivationFunction),
                MAX_LAYERS,  # for skip connection option
            ]
        )

    def _get_observation_space(self) -> spaces.Space:
        observation_space_vector: List[int] = []
        observation_space_vector.append(len(LayerType))
        observation_space_vector.append(len(OutChannels))
        observation_space_vector.append(len(KernelSize))
        observation_space_vector.append(len(Stride))
        observation_space_vector.append(len(PoolMode))
        observation_space_vector.append(len(ActivationFunction))
        observation_space_vector.append(len(LinearUnits))
        observation_space_vector.append(
            MAX_LAYERS
        )  # to denote a skip connection from a previous layer
        observation_space_vector *= MAX_LAYERS  # repeat for each layer

        return spaces.MultiDiscrete(observation_space_vector)

    def _get_observation(self) -> np.ndarray:
        flattened_obs = flatten_cnn_config(self.current_network_config, MAX_LAYERS)
        return np.array(flattened_obs, dtype=np.float32)

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

    def step(self, decision_logits: np.ndarray) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Execute one timestep within the environment.

        :Args:
        - decision_logits: The logits produced by the agent's policy network after masking. (represents decisions)

        :Returns:
        - tuple: (observation, reward, terminated, truncated, info)
        """

        self.info = {}

        decisions = transform_action_indices_to_decisions(decision_logits, MAX_LAYERS)
        if decisions.action_choice == StandardAction.NONE:  # Stop and evaluate
            reward = self.evaluate_architecture(self.current_network_config)
            terminated = True
            truncated = False
            self.actions_taken = 0  # Reset for next episode
        else:
            self.current_network_config += LayerConfig.from_decisions(decisions)
            reward = 0.05
            terminated = False
            truncated = False

        if not isinstance(reward, float):
            raise ValueError(
                f"{CustomEnv.__name__} does not support reward of type '{type(reward)}'"
            )

        obs = self._get_observation()
        self.actions_taken += 1

        return obs, reward, terminated, truncated, self.info

    def evaluate_architecture(self, new_architecture: NetworkConfig) -> float | dict[str, float]:
        """Evaluate the given architecture by training and testing it, returning the computed reward.

        :Args:
        - new_architecture: The CNN architecture to evaluate.

        :Returns:
        - float: The computed reward based on evaluation metrics.
        """
        architecture = Architecture(
            net_config=new_architecture,
            num_classes=self.data_importer.get_num_classes()[0],
            input_dimensions=self.dimensions,
        )

        trained_model, training_time = self.train_classifier(model=architecture)

        evaluated_metrics = self.evaluator.evaluate(trained_model, training_time=training_time)

        reward = self.reward_strategy.compute_reward(evaluated_metrics)

        self.tb_logger.log_evaluation(
            reward=reward,
            architecture=architecture,
            current_config=self.current_network_config,
            actions_taken=self.actions_taken,
            metrics=evaluated_metrics,
        )

        return reward

    def train_classifier(
        self,
        model: torch.nn.Module,
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

        model.to(self.device)
        optimizer = self.hyperparameters.get_optimizer(model.parameters())

        num_epochs = self.hyperparameters.training_epochs

        start_time = time.time()
        for epoch in range(num_epochs):
            progress = (epoch + 1) / num_epochs * 100
            with self.console.status(
                f"[bold blue]Training model on evaluation number '{self.tb_logger.evaluation_count}': Progress {int(progress)}%[/bold blue]"
            ):
                self.trainer.train(model, optimizer)

        end_time = time.time()
        training_time = end_time - start_time
        return model, training_time

    def _should_terminate(self) -> bool:
        # Termination is now handled in step() method
        return False

    def _has_truncated(self) -> bool:
        # Truncation is now handled in step() method
        return False

    def render(self):
        print(self._get_observation())

    def close(self):
        if self.render_mode == "console":
            pass
        else:
            raise NotImplementedError

    def get_action_mask(self) -> np.ndarray:
        """
        Returns a boolean mask for valid actions given the current environment state.
        This mask is used by MaskablePPO to know which actions are allowed globally.

        Its only purpose is to only allow ADD_LAYER if no layers exist yet.

        It is just necessary for masking to work even if it dont do much.
        """
        slices = get_logit_slices(max_layers=MAX_LAYERS)
        total_actions = slices.skip_connection.stop  # assuming this is the last slice

        mask = np.zeros(total_actions, dtype=bool)

        if self.current_network_config.layers == []:
            mask[slices.standard_actions[StandardAction.ADD_LAYER]] = True
        else:
            mask[slices.standard_actions.start : slices.standard_actions.stop] = True

        return mask
