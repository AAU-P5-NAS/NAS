from src.environment.reward.archive_pareto_dom import ProxyBaselines
from src.environment.metrics import TrainingFreeMetrics
from src.environment.reward.reward import Baselines, RewardStrategy, Weights


class TchebycheffRS(RewardStrategy):
    """
    weighted Tchebycheff scalarizing function: f_Tchb(x, w) = max_i * w_i |f_i(x) - z_star_i |

    """

    def __init__(self, weights: Weights, baselines: Baselines = Baselines()):
        super().__init__(weights, baselines)
        self.proxy_baselines = ProxyBaselines()

    def update_baselines(self, metrics: TrainingFreeMetrics):
        for proxy_name in ["synflow", "jacov", "snip", "complexity"]:
            value = getattr(metrics, proxy_name, None)
            if value is None:
                continue
            current_min, current_max = getattr(self.proxy_baselines, proxy_name) or (value, value)
            new_min = min(current_min, value)
            new_max = max(current_max, value)
            setattr(self.proxy_baselines, proxy_name, (new_min, new_max))

    def normalize_metrics(self, metrics: TrainingFreeMetrics) -> TrainingFreeMetrics:
        normalized = metrics.model_copy()
        for proxy_name in ["synflow", "jacov", "snip", "complexity"]:
            value = getattr(metrics, proxy_name, None)
            baseline = getattr(self.proxy_baselines, proxy_name)
            if value is None or baseline is None:
                continue
            min_val, max_val = baseline
            normalized_value = (value - min_val) / (max_val - min_val)
            setattr(normalized, proxy_name, normalized_value)
        return normalized

    def compute_reward(self, metrics: TrainingFreeMetrics) -> float:
        # self.update_baselines(metrics) # No need now, since baselines are initialized with fixed values
        weights = Weights.tchebycheffWeights().model_dump()
        proxy_baselines = self.proxy_baselines.model_dump()

        weighted_diffs = []
        for metric_name, weight in weights.items():
            if weight == 0:
                continue
            range_width = proxy_baselines[metric_name][1] - proxy_baselines[metric_name][0]
            diff = abs(
                getattr(metrics, metric_name, 0)
                - proxy_baselines[metric_name][0 if metric_name in ["complexity"] else 1]
            )
            normalized_diff = diff / range_width
            weighted_diff = weight * normalized_diff
            weighted_diffs.append(weighted_diff)
            # print("normalized_diff:", normalized_diff)
            # print("weighted_diff", weighted_diff)

        deviation = max(weighted_diffs, default=0.0)
        return 1 - deviation
