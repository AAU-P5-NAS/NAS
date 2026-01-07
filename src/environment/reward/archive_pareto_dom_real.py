import pickle
import os

from typing import List, Optional
import numpy as np
from pydantic import BaseModel
from src.environment.metrics import Metrics
from src.environment.reward.reward import Baselines, RewardStrategy, Weights
from src.utils.network_config import NetworkConfig
from src.utils.architecture import flatten_cnn_config

MAX_LAYERS = 7

ARCHIVE_DIR = "src/environment/reward/saved_archives"


class ArchiveEntry(BaseModel):
    arch: list[int]
    metrics: Metrics

class SortedLists(BaseModel):
    snip_sorted: List[ArchiveEntry]
    synflow_sorted: List[ArchiveEntry]
    jacov_sorted: List[ArchiveEntry]
    complexity_sorted: List[ArchiveEntry]
    ws_sorted: List[ArchiveEntry]


class ElitistArchive:
    def __init__(self):
        self.elites: List[ArchiveEntry] = []
        self.baselines = Baselines()
        self.epsilon = 1e-12  # small number to avoid division by zero

    def _normalize_metric(self, metric_name: str, value: float) -> float:
            """Normalize metric to [0, 1] range where 1 is better."""

            # Already normalized (higher is better)
            if metric_name in {"accuracy", "precision", "recall", "f1_score"}:
                return np.clip(value, 0.0, 1.0)

            # Test loss (lower is better, unbounded)
            elif metric_name == "test_loss":
                # Invert and normalize: 0 loss = 1.0, max_loss = 0.0
                normalized = 1.0 - np.clip(value / self.baselines.max_test_loss, 0.0, 1.0)
                return normalized

            # FLOPs (lower is better)
            elif metric_name == "flops":
                # normalized = 1.0 - np.clip(value / self.baselines.max_flops, 0.0, 1.0)
                normalized = 1 - np.clip(
                    np.log(value + 1) / np.log(self.baselines.max_flops + 1), 0.0, 1.0
                )
                return normalized

            # Runtime (lower is better)
            elif metric_name == "runtime":
                normalized = 1.0 - np.clip(value / self.baselines.max_runtime, 0.0, 1.0)
                return normalized

            # Architecture size (lower is better)
            elif metric_name == "architecture_size":
                normalized = 1.0 - np.clip(value / self.baselines.max_params, 0.0, 1.0)
                return normalized

            return 0.0


    def _eucl_distance(self, metrics1: Metrics, metrics2: Metrics) -> float:
        """Compute Euclidean distance between two metrics, skipping missing entries."""
        sum_sq = 0.0
        
        for metric in ["accuracy", "flops"]:
            val1 = getattr(metrics1, metric, None)
            val2 = getattr(metrics2, metric, None)
            if val1 is None or val2 is None:
                continue
            normalized_value = self._normalize_metric(metric, val1)
            normalized_value_2 = self._normalize_metric(metric, val2)
            sum_sq += (normalized_value - normalized_value_2) ** 2
        return sum_sq**0.5

    def _dominates(self, metrics_1: Metrics, metrics_2: Metrics) -> bool:
        """Check if new_arch dominates elite archs based on proxy metrics."""
        for proxy in ["accuracy", "flops"]:
            val1 = getattr(metrics_1, proxy, None)
            val2 = getattr(metrics_2, proxy, None)
            if val1 is None or val2 is None:
                continue
            if val1 < val2:
                return False
        return True

    def add(self, new_arch: ArchiveEntry):
        self.elites.append(new_arch)

    def remove(self, arch: ArchiveEntry):
        self.elites.remove(arch)

    def size(self) -> int:
        return len(self.elites)

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

class DominanceNoveltyRealRS(RewardStrategy):
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

    def compute_reward(self, metrics: Metrics, arch: NetworkConfig):
        dominance_score = self.compute_dominance(metrics=metrics, arch=arch)
        novelty_score = self.compute_novelty(metrics=metrics, arch=arch)

        total_reward = (
            self.dn_weights.dominance * dominance_score + self.dn_weights.novelty * novelty_score
        )

        return total_reward

    def compute_dominance(self, metrics: Metrics, arch: NetworkConfig):
        flattened_arch = flatten_cnn_config(arch, max_layers=MAX_LAYERS)  # less storage
        new_entry = ArchiveEntry(arch=flattened_arch, metrics=metrics)
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

    def compute_novelty(self, metrics: Metrics, arch: NetworkConfig):
        if not self.elite_archive.elites:
            return 0.0

        distances = [
            self.elite_archive._eucl_distance(
                metrics, elite.metrics
            )
            for elite in self.elite_archive.elites
        ]

        factor = 1 if arch in self.elite_archive.elites else -1
        novelty = factor * sum(distances) / len(self.elite_archive.elites)

        novelty_normalized = novelty / 4**0.5  # 4 proxy metrics
        return novelty_normalized

    def get_archive_size(self) -> int:
        return self.elite_archive.size()
