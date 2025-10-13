from pydantic import BaseModel
from src.classification_module.metrics import Metrics


class Weights(BaseModel):
    accuracy: float = 1.0  # Primary goal
    f1_score: float = 0.5  # Secondary goal
    efficiency: float = 0.2  # Penalize complexity


class RewardCalculator:
    def __init__(self, weights: Weights = Weights()):
        self.weights: Weights = weights

    def compute_reward(self, metrics: Metrics) -> float:
        """Compute reward focusing on accuracy with efficiency bonus."""

        # Get core metrics with defaults
        accuracy = getattr(metrics, "accuracy", 0.0)
        f1_score = getattr(metrics, "f1_score", 0.0)
        runtime = getattr(metrics, "runtime", 1.0)
        architecture_size = getattr(metrics, "architecture_size", 1)

        # Ensure valid ranges
        accuracy = max(0.0, min(1.0, accuracy))
        f1_score = max(0.0, min(1.0, f1_score))

        # Calculate base performance reward (0-1 scale)
        performance_reward = (
            self.weights.accuracy * accuracy + self.weights.f1_score * f1_score
        ) / (self.weights.accuracy + self.weights.f1_score)

        # Calculate efficiency penalty (0-1 scale, lower is better)
        # Penalize slow training and large architectures
        efficiency_penalty = min(1.0, (runtime / 10.0) + (architecture_size / 20.0))
        efficiency_bonus = (1.0 - efficiency_penalty) * self.weights.efficiency

        # Final reward: performance - efficiency penalty
        reward = performance_reward + efficiency_bonus * 0.1

        return max(0.0, reward)  # Ensure non-negative
