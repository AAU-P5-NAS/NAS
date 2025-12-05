from typing import Optional, overload
import numpy as np
import abc
import random

from pydantic import BaseModel, ConfigDict, field_validator
from src.environment.metrics import Metrics, TrainingFreeMetrics
from src.utils.network_config import NetworkConfig


class Weights(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    # Performance Metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    test_loss: float = 0.0

    # Cost/Efficiency Metrics
    flops: float = 0.0
    runtime: float = 0.0
    architecture_size: float = 0.0

    dominance: float = 0.0
    novelty: float = 0.0

    @classmethod
    def staticWeights(cls):
        return cls(
            accuracy=6.0,
            flops=2.0,
        )

    @classmethod
    def domnovWeights(cls):
        return cls(
            dominance=0.7,
            novelty=0.3,
        )

    @classmethod
    def dynamicWeightsSampler(cls):
        weights = {
            # Performance Metrics
            "accuracy": random.random(),
            # "precision": random.random(),
            # "recall": random.random(),
            # "f1_score": random.random(),
            # "test_loss": random.random(),
            # Cost/Efficiency Metrics
            "flops": random.random(),
            # "runtime": random.random(),
            # "architecture_size": random.random(),
        }

        total = sum(weights.values())

        for i in weights:
            weights[i] /= total
        return cls(**weights)

    def normalize(self) -> "Weights":
        """Return a new Weights instance with all values normalized to sum to 1."""
        total = sum(self.model_dump().values())
        if total == 0:
            raise ValueError("Total weight cannot be zero")
        normalized = {k: v / total for k, v in self.model_dump().items()}
        return Weights(**normalized)

    @field_validator("*")  # Validate weights
    def non_negative(cls, v, info):
        if v < 0:
            raise ValueError("Weights must be non-negative")
        return v


class Baselines(BaseModel):
    """Expected ranges for normalization"""

    max_flops: float = 5.9e9
    max_runtime: float = 300.0  # 5 minutes
    max_params: float = 1e7  # 10M parameters
    max_test_loss: float = 3.0  # Clip losses above this

    min_synflow: float = 1.0
    max_synflow: float = 1e6
    min_jacov: float = 0.0
    max_jacov: float = 1.0
    min_snip: float = 0.0
    max_snip: float = 500.0


class RewardStrategy(abc.ABC):
    def __init__(self, weights: Weights, baselines: Baselines = Baselines()):
        self.weights = weights
        self.baselines = baselines

    def _normalize_metric(self, metric_name: str, value: float) -> float:
        """Normalize metric to [0, 1] range where 1 is better."""

        # Already normalized (higher is better)
        if metric_name in {"accuracy", "precision", "recall", "f1_score"}:
            return np.clip(value, 0.0, 1.0)

        # Test loss (lower is better, unbounded)
        elif metric_name == "test_loss":
            # Invert and normalize: 0 loss = 1.0, max_loss = 0.0
            normalized = 1.0 - np.clip(value / self.baselines.max_test_loss, 0.0, 1.0)
            return normalized

        # FLOPs (lower is better)
        elif metric_name == "flops":
            # normalized = 1.0 - np.clip(value / self.baselines.max_flops, 0.0, 1.0)
            normalized = 1 - np.clip(
                np.log(value + 1) / np.log(self.baselines.max_flops + 1), 0.0, 1.0
            )
            return normalized

        # Runtime (lower is better)
        elif metric_name == "runtime":
            normalized = 1.0 - np.clip(value / self.baselines.max_runtime, 0.0, 1.0)
            return normalized

        # Architecture size (lower is better)
        elif metric_name == "architecture_size":
            normalized = 1.0 - np.clip(value / self.baselines.max_params, 0.0, 1.0)
            return normalized

        return 0.0

    # use overload for method overloading (cannot use abstract method for this)
    @overload
    def compute_reward(self, metrics: "Metrics") -> float | dict[str, float]: ...
    @overload
    def compute_reward(
        self, metrics: "TrainingFreeMetrics", arch_config: NetworkConfig
    ) -> float | dict[str, float]: ...

    def compute_reward(
        self,
        metrics: "Metrics | TrainingFreeMetrics",
        arch_config: Optional["NetworkConfig"] = None,
    ) -> float | dict[str, float]:
        raise NotImplementedError
