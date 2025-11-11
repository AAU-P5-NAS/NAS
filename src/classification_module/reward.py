from pydantic import BaseModel
from src.classification_module.metrics import Metrics
import numpy as np
import abc
import random


class Weights(BaseModel):
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

    @classmethod
    def staticWeights(cls):
        return cls(
            accuracy=6.0,
            flops=2.0,
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


class Baselines(BaseModel):
    """Expected ranges for normalization"""

    max_flops: float = 1e9  # 1 billion FLOPs
    max_runtime: float = 300.0  # 5 minutes
    max_params: float = 1e7  # 10M parameters
    max_test_loss: float = 3.0  # Clip losses above this


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

    @abc.abstractmethod
    def compute_reward(self, metrics: Metrics) -> float | dict[str, float]:
        """Compute reward based on provided metrics."""
        raise NotImplementedError


class WeightedSumRS(RewardStrategy):
    """
    A basic reward strategy that computes a weighted sum of normalized metrics.

    Weights and baselines can be customized via the Weights and Baselines classes.
    """

    def __init__(self, weights: Weights, baselines: Baselines = Baselines()):
        self.weights = weights
        self.baselines = baselines

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


class TchebycheffReward(RewardStrategy):
    """
    weighted Tchebycheff scalarizing function: f_Tchb(x, w) = max_i * w_i |f_i(x) - z_star_i |

    """

    def __init__(self, weights: Weights, baselines: Baselines = Baselines()):
        super().__init__(weights, baselines)

        # Compute ideal reference point z* from baselines
        self.z_star = {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0,
            "test_loss": 0.0,
            "flops": 0.0,
            "runtime": 0.0,
            "architecture_size": 0.0,
        }

    def compute_reward(self, metrics: Metrics) -> float:
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

        weighted_diffs = []
        for metric_name, weight in normalized_weights.items():
            if weight == 0:
                continue
            value = getattr(metrics, metric_name, None)
            if value is None:
                continue

            normalized_value = self._normalize_metric(metric_name, value)

            z_i_star = self.z_star.get(metric_name, 0.0)

            diff = weight * abs(normalized_value - z_i_star)
            weighted_diffs.append(diff)

        reward = max(weighted_diffs) if weighted_diffs else 0.0
        return reward


# _____________________________________
if __name__ == "__main__":
    metrics = Metrics(
        accuracy=0.8,
        precision=0.75,
        recall=0.7,
        f1_score=0.78,
        test_loss=0.3,
        flops=100,
        runtime=2.0,
        architecture_size=50,
    )

    weights = Weights.staticWeights()
    reward = WeightedSumRS(weights)
    print("Static W: ", weights, "\nreward: ", reward.compute_reward(metrics))
    print("")

    weights = Weights.dynamicWeightsSampler()
    reward = TchebycheffReward(weights)
    print("Dynamic W1: ", weights, "\nreward: ", reward.compute_reward(metrics))

    weights = Weights.dynamicWeightsSampler()
    reward = TchebycheffReward(weights)
    print("Dynamic W2: ", weights, "\nreward: ", reward.compute_reward(metrics))

    weights = Weights.dynamicWeightsSampler()
    reward = TchebycheffReward(weights)
    print("Dynamic W3: ", weights, "\nreward: ", reward.compute_reward(metrics))
