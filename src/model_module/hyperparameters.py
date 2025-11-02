"""Hyperparameter configuration classes for RL and Supervised Learning"""

from __future__ import annotations
from typing import Literal, Any
from pydantic import BaseModel, field_validator


class RLHyperParameters(BaseModel):
    """Hyperparameters for Reinforcement Learning agent"""

    learning_rate: float = 0.001
    learning_rate_choice: Literal[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3] | None = None

    def get_learning_rate(self) -> float:
        if self.learning_rate_choice is not None:
            return self.learning_rate_choice
        return self.learning_rate

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for compatibility"""
        return {
            "rl_lr": self.get_learning_rate(),
        }


class SLHyperParameters(BaseModel):
    """Hyperparameters for Supervised Learning (architecture training)"""

    learning_rate: float = 0.001
    learning_rate_min: float = 1e-4
    learning_rate_max: float = 1e-2

    momentum: float = 0.9
    momentum_min: float = 0.5
    momentum_max: float = 0.95

    batch_size: int = 64
    batch_size_choices: list[int] = [32, 64, 128, 256]

    optimizer_type: Literal["SGD", "Adam", "RMSprop"] = "SGD"

    training_epochs: int = 15
    training_epochs_choices: list[int] = [10, 15, 20, 25]

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Ensure batch size is valid"""
        valid_sizes = [32, 64, 128, 256]
        if v not in valid_sizes:
            raise ValueError(f"batch_size must be one of {valid_sizes}")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for compatibility"""
        return {
            "arch_lr": self.learning_rate,
            "arch_momentum": self.momentum,
            "batch_size": self.batch_size,
            "training_epochs": self.training_epochs,
            "optimizer_type": self.optimizer_type,
        }


class RewardWeightsConfig(BaseModel):
    """Configuration for reward function weights"""

    accuracy_weight_min: float = 4.0
    accuracy_weight_max: float = 10.0

    f1_weight_min: float = 5.0
    f1_weight_max: float = 15.0

    test_loss_weight_min: float = 3.0
    test_loss_weight_max: float = 8.0

    flops_weight_min: float = 1.0
    flops_weight_max: float = 4.0

    runtime_weight_min: float = 1.0
    runtime_weight_max: float = 5.0


class HyperparameterSearchSpace(BaseModel):
    """Search space configuration for hyperparameter optimization"""

    sl_hyperparameters: SLHyperParameters = SLHyperParameters()
    rl_hyperparameters: RLHyperParameters = RLHyperParameters()
    reward_weights: RewardWeightsConfig = RewardWeightsConfig()
