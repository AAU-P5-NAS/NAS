from pydantic import BaseModel
from src.classification_module.metrics import Metrics


class Weights(BaseModel):
    accuracy: float = 6.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 10.0
    test_loss: float = 5.0
    flops: float = 2.0
    runtime: float = 3.0
    architecture_size: float = 1.0


class RewardCalculator:
    def __init__(self, weights: Weights = Weights()):
        self.weights: Weights = weights

    def compute_reward(self, metrics: Metrics) -> float:
        """Compute reward as weighted combination of metrics."""

        reward = 0.0
        for weight in self.weights.model_dump().values():
            if weight < 0:
                raise ValueError("Weights must be non-negative")

        total_weight: float = sum(self.weights.model_dump().values())
        if total_weight != 1.0 and total_weight > 0:
            normalized_weights: dict[str, float] = {
                metric: weight / total_weight
                for metric, weight in self.weights.model_dump().items()
            }
        else:
            normalized_weights: dict[str, float] = self.weights.model_dump()

        for metric, weight in normalized_weights.items():
            value: float | int | None = getattr(metrics, metric, None)

            if value is None:
                continue

            if metric in {"flops", "runtime", "architecture_size"}:
                value = 1 / (1 + value)

            reward += weight * value

        return reward
