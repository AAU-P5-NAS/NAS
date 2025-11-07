from pydantic import BaseModel
from src.classification_module.metrics import Metrics
import numpy as np
import abc


class Weights(BaseModel):
    accuracy: float = 6.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 10.0
    test_loss: float = 5.0
    flops: float = 2.0
    runtime: float = 3.0
    architecture_size: float = 1.0


class Baselines(BaseModel):
    """Expected ranges for normalization"""

    max_flops: float = 1e9  # 1 billion FLOPs
    max_runtime: float = 300.0  # 5 minutes
    max_params: float = 1e7  # 10M parameters
    max_test_loss: float = 3.0  # Clip losses above this


class RewardStrategy(abc.ABC):
    @abc.abstractmethod
    def compute_reward(self, metrics: Metrics) -> float | dict[str, float]:
        """Compute reward based on provided metrics."""
        raise NotImplementedError


class FirstBasicRewardStrategy(RewardStrategy):
    """
    A basic reward strategy that computes a weighted sum of normalized metrics.

    Weights and baselines can be customized via the Weights and Baselines classes.
    """

    def __init__(self, weights: Weights = Weights(), baselines: Baselines = Baselines()):
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
            normalized = 1.0 - np.clip(value / self.baselines.max_flops, 0.0, 1.0)
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

    def compute_reward(self, metrics: Metrics) -> float:
        """
        Compute reward as weighted combination of normalized metrics.
        Returns a float reward in [0, 1].
        """

        # Validate weights
        for weight in self.weights.model_dump().values():
            if weight < 0:
                raise ValueError("Weights must be non-negative")

        # Normalize weights to sum to 1
        total_weight = sum(self.weights.model_dump().values())
        if total_weight == 0:
            raise ValueError("Total weight cannot be zero")

        normalized_weights = {
            metric: weight / total_weight for metric, weight in self.weights.model_dump().items()
        }

        # Calculate weighted sum of normalized metrics
        reward = 0.0
        for metric_name, weight in normalized_weights.items():
            if weight == 0:
                continue

            value = getattr(metrics, metric_name, None)
            if value is None:
                continue

            # Normalize metric to [0, 1] where 1 is better
            normalized_value = self._normalize_metric(metric_name, value)
            reward += weight * normalized_value

        return reward  # Now guaranteed in [0, 1]
