from src.environment.metrics import Metrics
from src.environment.reward.reward import Baselines, RewardStrategy, Weights


class WeightedSumRS(RewardStrategy):
    """
    A basic reward strategy that computes a weighted sum of normalized metrics.

    Weights and baselines can be customized via the Weights and Baselines classes.
    """

    def __init__(self, weights: Weights, baselines: Baselines = Baselines()):
        super().__init__(weights, baselines)

    def compute_reward(self, metrics: Metrics) -> float:
        """
        Compute reward as weighted combination of normalized metrics.
        Returns a float reward in [0, 1].
        """

        # Normalize weights to sum to 1
        normalized_weights = self.weights.normalize()

        # Calculate weighted sum of normalized metrics
        reward = 0.0
        for metric_name, weight in normalized_weights.model_dump().items():
            if weight == 0:
                continue

            value = getattr(metrics, metric_name, None)
            if value is None:
                continue

            # Normalize metric to [0, 1] where 1 is better
            normalized_value = self._normalize_metric(metric_name, value)
            reward += weight * normalized_value

        return reward  # Now guaranteed in [0, 1]