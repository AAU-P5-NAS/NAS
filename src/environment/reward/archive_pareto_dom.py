import pickle
import os

from typing import List, Optional, Tuple
from pydantic import BaseModel
from src.environment.metrics import TrainingFreeMetrics
from src.environment.reward.reward import Baselines, RewardStrategy, Weights
from src.utils.network_config import NetworkConfig
from src.utils.architecture import flatten_cnn_config

MAX_LAYERS = 7

ARCHIVE_DIR = "src/environment/reward/saved_archives"


class ProxyBaselines(BaseModel):
    synflow: Optional[Tuple[float, float]] = (0, 7.76e6)
    jacov: Optional[Tuple[float, float]] = (1.64e-19, 1.49e-5)
    snip: Optional[Tuple[float, float]] = (0, 2500)
    complexity: Optional[Tuple[float, float]] = (2.99e4, 1.75e7)


class ProxyBaselinesNone(BaseModel):
    synflow: Optional[Tuple[float, float]] = None
    jacov: Optional[Tuple[float, float]] = None
    snip: Optional[Tuple[float, float]] = None
    complexity: Optional[Tuple[float, float]] = None


class ArchiveEntry(BaseModel):
    arch: list[int]
    metrics: TrainingFreeMetrics


class SortedLists(BaseModel):
    snip_sorted: List[ArchiveEntry]
    synflow_sorted: List[ArchiveEntry]
    jacov_sorted: List[ArchiveEntry]
    complexity_sorted: List[ArchiveEntry]
    ws_sorted: List[ArchiveEntry]


class ElitistArchive:
    def __init__(self):
        self.elites: List[ArchiveEntry] = []
        self.proxy_baselines = ProxyBaselinesNone()
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
        return sum_sq**0.5

    def _dominates(self, metrics_1: TrainingFreeMetrics, metrics_2: TrainingFreeMetrics) -> bool:
        """Check if new_arch dominates elite archs based on proxy metrics."""
        for proxy in ["synflow", "jacov", "snip", "complexity"]:
            val1 = getattr(metrics_1, proxy, None)
            val2 = getattr(metrics_2, proxy, None)
            if val1 is None or val2 is None:
                continue
            if proxy in ["complexity", "jacov"]:  # minimize
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

    def save_archive(self):
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        existing_files = [
            f
            for f in os.listdir(ARCHIVE_DIR)
            if f.startswith("elite_archive") and f.endswith(".pkl")
        ]
        file_count = len(existing_files)

        filename = f"elite_archive_{file_count + 1}.pkl"
        with open(os.path.join(ARCHIVE_DIR, filename), "wb") as f:
            pickle.dump(self.elites, f)

        print(f"Archive saved to {filename}")

    def load_archive(self, archive_number: Optional[int] = None):
        if archive_number is None:
            # Load the latest archive
            existing_files = [
                f
                for f in os.listdir(ARCHIVE_DIR)
                if f.startswith("elite_archive") and f.endswith(".pkl")
            ]
            if not existing_files:
                raise FileNotFoundError("No archive files found in the directory.")
            latest_file = max(existing_files, key=lambda f: int(f.split("_")[2].split(".")[0]))
        else:
            # Load the specified archive
            latest_file = f"elite_archive_{archive_number}.pkl"
            if not os.path.exists(os.path.join(ARCHIVE_DIR, latest_file)):
                raise FileNotFoundError(f"Archive file {latest_file} not found.")

        with open(os.path.join(ARCHIVE_DIR, latest_file), "rb") as f:
            self.elites = pickle.load(f)
        print(f"Archive loaded from {latest_file}")
        return self.elites

    def sort_archs(self):
        elite_archs = ElitistArchive.load_archive(self)
        snip = elite_archs.copy()
        synflow = elite_archs.copy()
        jacov = elite_archs.copy()
        complexity = elite_archs.copy()
        ws = elite_archs.copy()

        snip.sort(reverse=True, key=lambda arch: arch.metrics.snip)  # type: ignore
        synflow.sort(reverse=True, key=lambda arch: arch.metrics.synflow)  # type: ignore
        jacov.sort(reverse=False, key=lambda arch: arch.metrics.jacov)  # type: ignore
        complexity.sort(reverse=False, key=lambda arch: arch.metrics.complexity)  # type: ignore

        baselines = {
            "synflow": (
                float(
                    synflow[-1].metrics.synflow if synflow[-1].metrics.synflow is not None else 0.0
                ),
                float(
                    synflow[0].metrics.synflow if synflow[0].metrics.synflow is not None else 0.0
                ),
            ),
            "jacov": (
                float(jacov[-1].metrics.jacov if jacov[-1].metrics.jacov is not None else 0.0),
                float(jacov[0].metrics.jacov if jacov[0].metrics.jacov is not None else 0.0),
            ),
            "snip": (
                float(snip[-1].metrics.snip if snip[-1].metrics.snip is not None else 0.0),
                float(snip[0].metrics.snip if snip[0].metrics.snip is not None else 0.0),
            ),
            "complexity": (
                float(
                    complexity[-1].metrics.complexity
                    if complexity[-1].metrics.complexity is not None
                    else 0.0
                ),
                float(
                    complexity[0].metrics.complexity
                    if complexity[0].metrics.complexity is not None
                    else 0.0
                ),
            ),
        }

        normalized_weights = Weights.tchebycheffWeights().model_dump()
        for arch in ws:
            sum = 0.0
            for metric_name, weight in normalized_weights.items():
                if weight == 0:
                    continue
                range_width = baselines[metric_name][1] - baselines[metric_name][0]
                diff = abs(getattr(arch.metrics, metric_name, 0) - baselines[metric_name][1])
                normalized_value = diff / range_width
                weighted_value = weight * normalized_value
                sum += weighted_value

            # print("Weighted sum for arch:", sum)
            setattr(arch.metrics, "accuracy", sum)

        ws.sort(reverse=True, key=lambda arch: arch.metrics.accuracy)  # type: ignore
        sorted_lists = SortedLists(
            snip_sorted=snip,
            synflow_sorted=synflow,
            jacov_sorted=jacov,
            complexity_sorted=complexity,
            ws_sorted=ws,
        )
        return sorted_lists


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

    def compute_reward(self, metrics: TrainingFreeMetrics, arch: NetworkConfig):
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
            self.elite_archive._eucl_distance(
                normalized_metrics, self.elite_archive.normalize_metrics(elite.metrics)
            )
            for elite in self.elite_archive.elites
        ]

        factor = 1 if arch in self.elite_archive.elites else -1
        novelty = factor * sum(distances) / len(self.elite_archive.elites)

        novelty_normalized = novelty / 4**0.5  # 4 proxy metrics
        return novelty_normalized

    def get_archive_size(self) -> int:
        return self.elite_archive.size()
