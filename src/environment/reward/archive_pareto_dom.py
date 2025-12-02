from typing import List, Optional, Tuple
from pydantic import BaseModel
from src.environment.metrics import TrainingFreeMetrics
from src.environment.reward.reward import Baselines, RewardStrategy, Weights
from src.utils.network_config import NetworkConfig
from src.utils.architecture import flatten_cnn_config

MAX_LAYERS = 7


class ProxyBaselines(BaseModel):
    synflow: Optional[Tuple[float, float]] = None
    jacov: Optional[Tuple[float, float]] = None
    snip: Optional[Tuple[float, float]] = None
    complexity: Optional[Tuple[float, float]] = None


class ArchiveEntry(BaseModel):
    arch: list[int]
    metrics: TrainingFreeMetrics

class ElitistArchive:
    def __init__(self):
        self.elites: List[ArchiveEntry] = []
        self.proxy_baselines = ProxyBaselines()
        self.epsilon = 1e-12  # small number to avoid division by zero

    def _eucl_distance(self, metrics1: TrainingFreeMetrics, metrics2: TrainingFreeMetrics) -> float:
        """Compute Euclidean distance between two metrics, skipping missing entries."""
        sum_sq = 0.0
        for proxy in ["synflow", "jacov", "snip", "complexity"]:
            val1 = getattr(metrics1, proxy, None)
            val2 = getattr(metrics2, proxy, None)
            if val1 is None or val2 is None:
                continue
            sum_sq += (val1 - val2) ** 2
        return sum_sq ** 0.5

    def _dominates(self, metrics_1: TrainingFreeMetrics, metrics_2: TrainingFreeMetrics) -> bool:
        """Check if new_arch dominates elite archs based on proxy metrics."""
        for proxy in ["synflow", "jacov", "snip", "complexity"]:
            val1 = getattr(metrics_1, proxy, None)
            val2 = getattr(metrics_2, proxy, None)
            if val1 is None or val2 is None:
                continue
            if proxy == "complexity":  # minimize
                if val1 > val2:
                    return False
            else:  # maximize
                if val1 < val2:
                    return False
        return True

    def add(self, new_arch: ArchiveEntry):
        self.update_baselines(new_arch.metrics)
        self.elites.append(new_arch)

    def remove(self, arch: ArchiveEntry):
        self.elites.remove(arch)

    def size(self) -> int:
        return len(self.elites)

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
            normalized_value = (value - min_val) / (max_val - min_val + self.epsilon)
            setattr(normalized, proxy_name, normalized_value)
        return normalized

    def check_domination(self, new_entry: ArchiveEntry) -> tuple[bool, int, List[ArchiveEntry]]:
        num_dominations = 0
        dominated_elites: List[ArchiveEntry] = []

        for elite in self.elites:
            if self._dominates(elite.metrics, new_entry.metrics):
                num_dominations += 1
            elif self._dominates(new_entry.metrics, elite.metrics):
                dominated_elites.append(elite)

        return num_dominations > 0, num_dominations, dominated_elites
    


class DominanceNoveltyRS(RewardStrategy):
    """
    A reward strategy based on Pareto dominance and novelty.

    Rewards architectures that are either high-performing on multiple metrics (Dominance)
    or significantly different from what’s been seen before (Novelty),
    steering the search toward a diverse Pareto front.
    """

    def __init__(self, weights, baselines: Baselines = Baselines(), domnov_weights=(0.5, 0.5)):
        super().__init__(weights, baselines)
        self.elite_archive = ElitistArchive()
        self.dn_weights = Weights().domnovWeights()

    def compute_reward(
        self, metrics: TrainingFreeMetrics, arch: NetworkConfig
    ):
        dominance_score = self.compute_dominance(proxy_metrics=metrics, arch=arch)
        novelty_score = self.compute_novelty(proxy_metrics=metrics, arch=arch)

        total_reward = (
            self.dn_weights.dominance * dominance_score + self.dn_weights.novelty * novelty_score
        )

        return total_reward

    def compute_dominance(self, proxy_metrics: TrainingFreeMetrics, arch: NetworkConfig):
        flattened_arch = flatten_cnn_config(arch, max_layers=MAX_LAYERS)  # less storage
        new_entry = ArchiveEntry(arch=flattened_arch, metrics=proxy_metrics)
        is_dominated, num_dominations, elite_archs_dominated = self.elite_archive.check_domination(
            new_entry
        )

        if not is_dominated:
            self.elite_archive.add(new_entry)
            for dominated_arch in elite_archs_dominated:
                self.elite_archive.remove(dominated_arch)

        # Reward inversely proportional to number of dominations
        dominance_score = 1.0 / (1 + num_dominations)
        return dominance_score

    def compute_novelty(self, proxy_metrics: TrainingFreeMetrics, arch: NetworkConfig):
        if not self.elite_archive.elites:
            return 0.0
        
        normalized_metrics = self.elite_archive.normalize_metrics(proxy_metrics)
        distances = [
            self.elite_archive._eucl_distance(normalized_metrics, self.elite_archive.normalize_metrics(elite.metrics))
            for elite in self.elite_archive.elites
        ]
        
        factor = 1 if arch in self.elite_archive.elites else -1
        novelty = factor * sum(distances) / len(self.elite_archive.elites)

        novelty_normalized = novelty / 4**0.5 # 4 proxy metrics
        return novelty_normalized
    

    def get_archive_size(self) -> int:
        return self.elite_archive.size()
