from typing import List, Optional, Tuple
from pydantic import BaseModel
from src.environment.metrics import TrainingFreeMetrics
from src.environment.reward.reward import Baselines, RewardStrategy, Weights
from src.utils.network_config import NetworkConfig
from src.utils.architecture import flatten_cnn_config, unflatten_cnn_config

MAX_LAYERS = 10


class ProxyBaselines(BaseModel):
    synflow: Optional[Tuple[float, float]] = None
    jacov: Optional[Tuple[float, float]] = None
    snip: Optional[Tuple[float, float]] = None
    complexity: Optional[Tuple[float, float]] = None


class ArchiveEntry(BaseModel):
    arch: list[int]
    metrics: TrainingFreeMetrics

    def dominates(self, other: "ArchiveEntry") -> bool:
        """Check if self dominates other based on proxy metrics."""
        for proxy, self_value in self.metrics.model_dump().items():
            other_value = other.metrics.model_dump().get(proxy)
            if other_value is None:
                continue
            if self_value < other_value:
                return False  # other is better in this proxy
        return True  # self dominates other

    def eucl_distance(self, other_metrics: TrainingFreeMetrics) -> float:
        """Calculate Euclidean distance between self and other based on proxy metrics."""
        sum_sq = 0.0

        for proxy, self_value in self.metrics.model_dump().items():
            other_value = other_metrics.model_dump().get(proxy)
            if other_value is None:
                continue
            sum_sq += (self_value - other_value) ** 2
        return sum_sq**0.5


class ElitistArchive:
    def __init__(self):
        self.elites: List[ArchiveEntry] = []
        self.proxy_baselines = ProxyBaselines()
        self.epsilon = 1e-12  # small number to avoid division by zero

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

    def add(self, new_arch: ArchiveEntry):
        self.update_baselines(new_arch.metrics)
        print("BASELINES: ", self.proxy_baselines)
        self.elites.append(new_arch)

    def remove(self, arch: ArchiveEntry):
        self.elites.remove(arch)

    def size(self) -> int:
        return len(self.elites)

    def check_domination(self, new_arch: ArchiveEntry) -> tuple[bool, int, List[ArchiveEntry]]:
        num_dominations = 0
        elite_archs_dominated: List[ArchiveEntry] = []

        for elite_arch in self.elites:
            if elite_arch.dominates(new_arch):
                num_dominations += 1
            elif new_arch.dominates(elite_arch):
                elite_archs_dominated.append(elite_arch)

        return num_dominations > 0, num_dominations, elite_archs_dominated

    def calculate_novelty(self, arch: ArchiveEntry):
        if self.size() == 0:
            return 0.0

        arch_normalized = arch.model_copy()
        arch_normalized.metrics = self.normalize_metrics(arch.metrics)

        if arch in self.elites:
            return (
                1
                / self.size()
                * sum(
                    arch_normalized.eucl_distance(self.normalize_metrics(elite.metrics))
                    for elite in self.elites
                )
            )
        else:
            return -(1 / self.size()) * sum(
                arch_normalized.eucl_distance(self.normalize_metrics(elite.metrics))
                for elite in self.elites
            )


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

    def compute_dominance(self, proxy_metrics: TrainingFreeMetrics, arch: NetworkConfig) -> float:
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

    def compute_novelty(self, proxy_metrics: TrainingFreeMetrics, arch: NetworkConfig) -> float:
        new_entry = ArchiveEntry(
            arch=flatten_cnn_config(arch, max_layers=MAX_LAYERS), metrics=proxy_metrics
        )
        novelty = self.elite_archive.calculate_novelty(new_entry)
        novelty /= 4**0.5
        return novelty

    def compute_reward(
        self, metrics: TrainingFreeMetrics, arch: NetworkConfig
    ) -> float | dict[str, float]:
        dominance_score = self.compute_dominance(proxy_metrics=metrics, arch=arch)
        novelty_score = self.compute_novelty(proxy_metrics=metrics, arch=arch)

        """ print("Dominance Score:", dominance_score)
        print("Novelty Score:", novelty_score)
        print("self.elite_archive size:", self.elite_archive.size()) """

        total_reward = (
            self.dn_weights.dominance * dominance_score + self.dn_weights.novelty * novelty_score
        )

        return total_reward

    def get_size_of_archive(self) -> int:
        return self.elite_archive.size()

"""     def get_highest_value_arch(self) -> NetworkConfig:
        self.elite_archive.rank_aggregate_sort()
        best = self.elite_archive.elites[0]
        arch = unflatten_cnn_config(best.arch, max_layers=10)
        return arch
 """