from typing import Any
from src.environment.metrics import TrainingFreeMetrics
from src.environment.reward.reward import RewardStrategy
from src.utils.network_config import NetworkConfig


class DominanceNoveltyRS(RewardStrategy):
    """
    A reward strategy based on Pareto dominance and novelty.

    Rewards architectures that are either high-performing on multiple metrics (Dominance) 
    or significantly different from what’s been seen before (Novelty), 
    steering the search toward a diverse Pareto front.
    """
    def __init__(self, weights, baselines=None):
        super().__init__(weights, baselines)
        self.elite_archive = []  # Store elite architectures for novelty comparison



    def compute_dominance(self, params: Any) -> float:
        # Placeholder for dominance computation logic
        return 0.0

    def compute_novelty(self, params: Any) -> float:
        # Placeholder for novelty computation logic
        return 0.0

    def compute_reward(self, metrics: TrainingFreeMetrics, arch: NetworkConfig) -> float | dict[str, float]:

        # Placeholder implementation
        # Actual implementation would compute dominance and novelty scores
        dominance_score = self.compute_dominance(metrics)  # Compute based on metrics
        novelty_score = self.compute_novelty(metrics)    # Compute based on difference in architecture from k-nearest neighbors

        # Combine scores (weights can be adjusted)
        total_reward = 0.5 * dominance_score + 0.5 * novelty_score
        return total_reward
