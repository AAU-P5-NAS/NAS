from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from stable_baselines3.common.logger import Logger
from torch.utils.tensorboard import SummaryWriter
import torch
from rich.console import Console

from src.environment.metrics import Metrics, TrainingFreeMetrics
from src.utils.network_config import  NetworkConfig
from src.utils.layer_config import LayerConfig


@dataclass
class ArchitectureCacheEntry:
    architecture: list[LayerConfig]
    metrics: Metrics
    reward: float


class BestFiftyArchitecturesCache:
    # Sorted list of best fifty architectures based on accuracy
    cache: list[ArchitectureCacheEntry] = []

    def add_entry_if_needed(
        self,
        architecture: list[LayerConfig],
        metrics: Metrics,
        reward: float,
        tensorboard_logger: TensorboardLogger,
    ):
        entry = ArchitectureCacheEntry(architecture=architecture, metrics=metrics, reward=reward)

        # If cache has less than 50 entries, add directly
        if len(self.cache) < 50:
            self.cache.append(entry)
            self.cache.sort(key=lambda x: x.reward or 0, reverse=True)

        else:
            # Check if new entry is better than the worst in cache
            worst_entry = self.cache[-1]
            if (metrics.accuracy or 0) > (worst_entry.metrics.accuracy or 0):
                self.cache[-1] = entry
                self.cache.sort(key=lambda x: x.reward or 0, reverse=True)

        self._write_to_file(tensorboard_logger)

    def _write_to_file(self, tensorboard_logger: TensorboardLogger):
        with open("best_fifty_architectures.txt", "w") as f:
            for i, entry in enumerate(self.cache):
                f.write(f"Architecture Rank {i + 1}:\n")
                f.write(f"Metrics: {entry.metrics}\n")
                f.write(
                    tensorboard_logger.get_layers_as_str(entry.architecture, is_for_console=False)
                )
                f.write("\n" + "=" * 80 + "\n\n")


class TensorboardLogger:
    best_fifty_cache = BestFiftyArchitecturesCache()

    def __init__(
        self,
        logger: Logger,
        writer: SummaryWriter,
        log_interval: int = 1,
        log_folder: str = "tensorboard_logs/",
    ):
        self.writer = writer
        self.logger = (
            logger
            if logger is not None
            else Logger(folder=log_folder, output_formats=[self.tb_format])
        )
        self.log_interval = log_interval
        self.dimensions = (3, 32, 32)  # for cifar10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.console = Console()

        self.evaluation_count: int = 0
        self.sum_reward_float: float = 0.0
        self.sum_reward_dict: dict[str, float] = {}
        self.sum_accuracy: float = 0.0

        self.newest_reward: Optional[float] = None
        self.newest_actions_taken: Optional[int] = None
        self.newest_metrics: Optional[Metrics] = None
        self.newest_architecture: Optional[torch.nn.Module] = None
        self.current_config: Optional[NetworkConfig] = None

    def attach_logger(self, logger: Logger, writer: SummaryWriter):
        self.logger = logger
        self.writer = writer

    def update(
        self,
        reward: Optional[float] = None,
        actions_taken: Optional[int] = None,
        metrics: Optional[Metrics] = None,
        proxy_metrics: Optional[TrainingFreeMetrics] = None,
        architecture: torch.nn.Module | None = None,
        current_config: NetworkConfig | None = None,
    ):
        """Update the logger's latest state."""

        if reward is not None:
            self.newest_reward = reward
        if actions_taken is not None:
            self.newest_actions_taken = actions_taken
        if metrics is not None:
            self.newest_metrics = metrics
        if architecture is not None:
            self.newest_architecture = architecture
        if current_config is not None:
            self.current_config = current_config

        if self.evaluation_count % self.log_interval == 0:
            self.log_to_tensorboard(proxy_metrics=proxy_metrics)

    def log_to_tensorboard(self, proxy_metrics: Optional[TrainingFreeMetrics] = None):
        """Write current stored metrics to TensorBoard."""
        if self.logger is None or self.writer is None:
            raise RuntimeError("Logger not attached")

        def record_optional(name: str, value: Optional[float]):
            if value is not None:
                self.logger.record(name, value)

        record_optional("Custom/Reward", self.newest_reward)
        record_optional("Custom/Actions Taken", self.newest_actions_taken)
        if self.newest_metrics is not None:
            record_optional("Custom/Test Loss", self.newest_metrics.test_loss)
            record_optional("Custom/Accuracy", self.newest_metrics.accuracy)
            record_optional("Custom/Precision", self.newest_metrics.precision)
            record_optional("Custom/Recall", self.newest_metrics.recall)
            record_optional("Custom/F1 Score", self.newest_metrics.f1_score)
            record_optional("Custom/FLOPs", self.newest_metrics.flops)
            record_optional("Custom/Runtime", self.newest_metrics.runtime)
            record_optional("Custom/Architecture Size", self.newest_metrics.architecture_size)
        
        if proxy_metrics is not None:
            record_optional("Custom Proxy/Synflow", proxy_metrics.synflow)
            record_optional("Custom Proxy/Jacov", proxy_metrics.jacov)
            record_optional("Custom Proxy/Snip", proxy_metrics.snip)
            record_optional("Custom Proxy/Complexity", proxy_metrics.complexity)

        self.logger.dump(step=self.evaluation_count)

        """ if self.newest_architecture is not None:
            channels, h, w = self.dimensions
            self.writer.add_graph(
                self.newest_architecture,
                torch.zeros(1, channels, h, w).to(device=self.device),
            ) """

    def log_evaluation(
        self,
        reward: float | dict[str, float],        
        architecture: torch.nn.Module | None,
        current_config: NetworkConfig | None,
        actions_taken: Optional[int] = None,      
        metrics: Optional[Metrics] = None,  
        proxy_metrics: Optional[TrainingFreeMetrics] = None,

    ):
        """Update state, print to console, and log to TensorBoard."""

        self.evaluation_count += 1

        # Track averages first
        if isinstance(reward, float):
            self.sum_reward_float += reward
            avg_reward = self.sum_reward_float / self.evaluation_count
            reward_to_store = reward
        elif isinstance(reward, dict):
            for k, v in reward.items():
                self.sum_reward_dict.setdefault(k, 0.0)
                self.sum_reward_dict[k] += v
            avg_reward = {k: v / self.evaluation_count for k, v in self.sum_reward_dict.items()}
            reward_to_store = sum(reward.values())

        # Then update logger state (this calls log_to_tensorboard)
        self.update(
            reward=reward_to_store,
            metrics=metrics,
            proxy_metrics=proxy_metrics,
            architecture=architecture,
            current_config=current_config,
            actions_taken=actions_taken,
        )
        if proxy_metrics is not None:
            return

        if metrics is None:
            return
        
        avg_accuracy = -1
        if metrics.accuracy is not None:
            self.sum_accuracy += metrics.accuracy
            avg_accuracy = self.sum_accuracy / self.evaluation_count

        # Print to console
        self.console.print(
            f"[bold green]Evaluation {self.evaluation_count} (actions {actions_taken})[/bold green]"
        )
        self.console.print(f"[bold blue]Reward: {reward}, Avg: {avg_reward}[/bold blue]")
        if metrics.accuracy is not None:
            self.console.print(
                f"[bold blue]Accuracy: {metrics.accuracy}, Avg: {avg_accuracy:.4f}[/bold blue]"
            )

        self.print_layers(current_config.layers if current_config else [])

        if current_config and self.newest_reward:
            self.best_fifty_cache.add_entry_if_needed(
                architecture=current_config.layers,
                metrics=metrics,
                reward=self.newest_reward,
                tensorboard_logger=self,
            )

        PRINT_EVERY_N = 50
        if self.evaluation_count % PRINT_EVERY_N == 0:
            self.console.print(
                f"[bold cyan]Average reward over last {PRINT_EVERY_N} evals: {avg_reward}[/bold cyan]"
            )
            self.console.print(
                f"[bold cyan]Average accuracy over last {PRINT_EVERY_N} evals: {avg_accuracy:.4f}[/bold cyan]"
            )
            self.sum_reward_float = 0.0
            self.sum_reward_dict = {}
            self.sum_accuracy = 0.0

    def print_layers(self, layers: list[LayerConfig]):
        """
        Print details of each layer to the console.
        Args:
            layers (list): List of layer objects.
        """

        self.console.print(self.get_layers_as_str(layers, is_for_console=True))

    def get_layers_as_str(self, layers: list[LayerConfig], is_for_console: bool) -> str:
        layers_str = ""
        indent = "    "

        if is_for_console:
            layers_str += "[bold yellow]Architecture:[/bold yellow]"

            for i, layer in enumerate(layers):
                if hasattr(layer, "layer_type") and layer.layer_type.name == "CONV":
                    layers_str += f"\n{indent}[bold yellow]Layer {i}:[/bold yellow] {layer.layer_type.name} - OutChannels: {layer.out_channels.name}, Kernel Size: {layer.kernel_size.name}, Stride: {layer.stride.name}, Pool Mode: {layer.pool_mode.name}, Activation: {layer.activation.name}"

                elif hasattr(layer, "layer_type") and layer.layer_type.name == "LINEAR":
                    layers_str += f"\n{indent}[bold yellow]Layer {i}:[/bold yellow] {layer.layer_type.name} - LinearUnits: {layer.linear_units.name}, Activation: {layer.activation.name}"

                elif hasattr(layer, "layer_type") and layer.layer_type.name == "POOL":
                    layers_str += f"\n{indent}[bold yellow]Layer {i}:[/bold yellow] {layer.layer_type.name} - Pool Mode: {layer.pool_mode.name}, Kernel Size: {layer.kernel_size.name}, Stride: {layer.stride.name} , Pool Mode: {layer.pool_mode.name}, Activation: {layer.activation.name}"

        else:
            for i, layer in enumerate(layers):
                if hasattr(layer, "layer_type") and layer.layer_type.name == "CONV":
                    layers_str += f"\nLayer {i}: {layer.layer_type.name} - OutChannels: {layer.out_channels.name}, Kernel Size: {layer.kernel_size.name}, Stride: {layer.stride.name}, Pool Mode: {layer.pool_mode.name}, Activation: {layer.activation.name}\n"

                elif hasattr(layer, "layer_type") and layer.layer_type.name == "LINEAR":
                    layers_str += f"\nLayer {i}: {layer.layer_type.name} - LinearUnits: {layer.linear_units.name}, Activation: {layer.activation.name}\n"

                elif hasattr(layer, "layer_type") and layer.layer_type.name == "POOL":
                    layers_str += f"\nLayer {i}: {layer.layer_type.name} - Pool Mode: {layer.pool_mode.name}, Kernel Size: {layer.kernel_size.name}, Stride: {layer.stride.name} , Pool Mode: {layer.pool_mode.name}, Activation: {layer.activation.name}\n"

        return layers_str

def get_layers_as_str(layers: list[LayerConfig], is_for_console: bool) -> str:
    layers_str = ""
    indent = "    "

    if is_for_console:
        layers_str += "[bold yellow]Architecture:[/bold yellow]"

        for i, layer in enumerate(layers):
            if hasattr(layer, "layer_type") and layer.layer_type.name == "CONV":
                layers_str += f"\n{indent}[bold yellow]Layer {i}:[/bold yellow] {layer.layer_type.name} - OutChannels: {layer.out_channels.name}, Kernel Size: {layer.kernel_size.name}, Stride: {layer.stride.name}, Pool Mode: {layer.pool_mode.name}, Activation: {layer.activation.name}"

            elif hasattr(layer, "layer_type") and layer.layer_type.name == "LINEAR":
                layers_str += f"\n{indent}[bold yellow]Layer {i}:[/bold yellow] {layer.layer_type.name} - LinearUnits: {layer.linear_units.name}, Activation: {layer.activation.name}"

            elif hasattr(layer, "layer_type") and layer.layer_type.name == "POOL":
                layers_str += f"\n{indent}[bold yellow]Layer {i}:[/bold yellow] {layer.layer_type.name} - Pool Mode: {layer.pool_mode.name}, Kernel Size: {layer.kernel_size.name}, Stride: {layer.stride.name} , Pool Mode: {layer.pool_mode.name}, Activation: {layer.activation.name}"

    else:
        for i, layer in enumerate(layers):
            if hasattr(layer, "layer_type") and layer.layer_type.name == "CONV":
                layers_str += f"\nLayer {i}: {layer.layer_type.name} - OutChannels: {layer.out_channels.name}, Kernel Size: {layer.kernel_size.name}, Stride: {layer.stride.name}, Pool Mode: {layer.pool_mode.name}, Activation: {layer.activation.name}\n"

            elif hasattr(layer, "layer_type") and layer.layer_type.name == "LINEAR":
                layers_str += f"\nLayer {i}: {layer.layer_type.name} - LinearUnits: {layer.linear_units.name}, Activation: {layer.activation.name}\n"

            elif hasattr(layer, "layer_type") and layer.layer_type.name == "POOL":
                layers_str += f"\nLayer {i}: {layer.layer_type.name} - Pool Mode: {layer.pool_mode.name}, Kernel Size: {layer.kernel_size.name}, Stride: {layer.stride.name} , Pool Mode: {layer.pool_mode.name}, Activation: {layer.activation.name}\n"

    return layers_str

class NoOpLogger:
    evaluation_count: int = 0

    def update(self, *args, **kwargs):
        pass

    def log_evaluation(self, *args, **kwargs):
        pass

    def attach_logger(self, *args, **kwargs):
        pass

    def print_layers(self, *args, **kwargs):
        pass
