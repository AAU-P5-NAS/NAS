from pydantic import BaseModel
from src.classification_module.metrics import Metrics


class Weights(BaseModel):
    accuracy: float = 1.0
    f1_score: float = 0.5
    runtime_penalty: float = 0.1
    size_penalty: float = 0.05


class RewardCalculator:
    def __init__(self, weights: Weights = Weights()):
        self.weights: Weights = weights

    def compute_reward(self, metrics: Metrics) -> float:
        """
        Compute reward with exponential scaling for high accuracy.

        Reward structure:
        - Random guessing (~0.038): -10
        - Low accuracy (<0.3): -5 to 0
        - Medium accuracy (0.3-0.7): 0 to 5
        - Good accuracy (0.7-0.85): 5 to 12
        - Excellent accuracy (0.85-0.95): 12 to 25
        - Outstanding accuracy (>0.95): 25+
        """

        # Get core metrics with defaults
        accuracy = getattr(metrics, "accuracy", 0.0)
        f1_score = getattr(metrics, "f1_score", 0.0)
        runtime = getattr(metrics, "runtime", 1.0)
        architecture_size = getattr(metrics, "architecture_size", 100000)

        # Ensure valid ranges
        accuracy = max(0.0, min(1.0, accuracy))
        f1_score = max(0.0, min(1.0, f1_score))

        # Strong penalty for essentially random performance (26 classes = ~3.8% random)
        if accuracy < 0.1:
            return -10.0

        # Combined performance metric
        performance_score = (
            self.weights.accuracy * accuracy + self.weights.f1_score * f1_score
        ) / (self.weights.accuracy + self.weights.f1_score)

        # QUADRATIC scaling: rewards high accuracy more, but not explosively
        # Using: reward = 10 * performance^2
        # This makes 0.85->0.93 more valuable than 0.70->0.78, but smoothly
        # Scale factor adjusted so 0.95 accuracy ≈ 9 reward
        performance_reward = 10.0 * (performance_score**2)

        # Runtime penalty (penalize slow models)
        # Target: ~2s is good, >10s starts getting penalized
        runtime_penalty = 0.0
        if runtime > 2.0:
            # Softer penalty - don't penalize training time too heavily during search
            runtime_penalty = self.weights.runtime_penalty * ((runtime - 2.0) / 10.0)

        # Architecture size penalty (penalize overly complex models)
        # Normalize by typical size (e.g., 50k parameters is reasonable)
        size_penalty = self.weights.size_penalty * (architecture_size / 50000.0)

        # Final reward with penalties
        reward = performance_reward - runtime_penalty - size_penalty

        return reward
