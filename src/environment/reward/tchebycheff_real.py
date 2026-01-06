from src.environment.metrics import Metrics
from src.environment.reward.reward import Baselines, RewardStrategy, Weights


class TchebycheffRealRS(RewardStrategy):
    """
    weighted Tchebycheff scalarizing function: f_Tchb(x, w) = max_i * w_i |f_i(x) - z_star_i |

    """

    def __init__(self, weights: Weights, baselines: Baselines = Baselines()):
        super().__init__(weights, baselines)

        # Compute ideal reference point z* from baselines
        self.z_star = {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "test_loss": 0.0,
            "flops": 0.0,
            "runtime": 0.0,
            "architecture_size": 0.0,
        }

    def compute_reward(self, metrics: Metrics) -> float:
        # Normalize weights to sum to 1
        normalized_weights = self.weights.normalize()

        weighted_diffs = []
        for metric_name, weight in normalized_weights.model_dump().items():
            if weight == 0:
                continue
            value = getattr(metrics, metric_name, None)
            if value is None:
                continue

            normalized_value = self._normalize_metric(metric_name, value)

            z_i_star = self.z_star.get(metric_name, 0.0)

            diff = weight * abs(normalized_value - z_i_star)
            weighted_diffs.append(diff)

        deviation = max(weighted_diffs) if weighted_diffs else 0.0
        return 1 - deviation
