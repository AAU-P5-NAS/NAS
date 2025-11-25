from typing import Optional
from pydantic import BaseModel, ConfigDict
from stable_baselines3.common.logger import Logger
from torch.utils.tensorboard import SummaryWriter
import torch
from rich.console import Console

from src.environment.metrics import Metrics
from src.utils.network_config import  NetworkConfig
from src.utils.layer_config import LayerConfig


class LogData(BaseModel):
    reward: Optional[float] = None
    actions_taken: Optional[int] = None
    metrics: Optional[Metrics] = None
    architecture: Optional[torch.nn.Module] = None
    step: int
    model_config = ConfigDict(arbitrary_types_allowed=True)


class TensorboardLogger:
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
        architecture: torch.nn.Module | None = None,
        current_config: NetworkConfig | None = None,
    ):
        """Update the logger's latest state."""
        print(
            "metrics received:",
        )
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
            self.log_to_tensorboard()

    def log_to_tensorboard(self):
        """Write current stored metrics to TensorBoard."""
        if self.logger is None or self.writer is None:
            raise RuntimeError("Logger not attached")

        def record_optional(name: str, value: Optional[float]):
            if value is not None:
                self.logger.record(name, value)

        record_optional("Custom/Reward", self.newest_reward)
        record_optional("Custom/Actions Taken", self.newest_actions_taken)
        print("self metrics: ", self.newest_metrics)
        if self.newest_metrics is not None:
            record_optional("Custom/Test Loss", self.newest_metrics.test_loss)
            record_optional("Custom/Accuracy", self.newest_metrics.accuracy)
            record_optional("Custom/Precision", self.newest_metrics.precision)
            record_optional("Custom/Recall", self.newest_metrics.recall)
            record_optional("Custom/F1 Score", self.newest_metrics.f1_score)
            record_optional("Custom/FLOPs", self.newest_metrics.flops)
            record_optional("Custom/Runtime", self.newest_metrics.runtime)
            record_optional("Custom/Architecture Size", self.newest_metrics.architecture_size)

        self.logger.dump(step=self.evaluation_count)

        if self.newest_architecture is not None:
            channels, h, w = self.dimensions
            self.writer.add_graph(
                self.newest_architecture,
                torch.zeros(1, channels, h, w).to(device=self.device),
            )

    def log_evaluation(
        self,
        metrics: Metrics,
        reward: float | dict[str, float],
        architecture: torch.nn.Module | None,
        current_config: NetworkConfig | None,
        actions_taken: Optional[int] = None,
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
            architecture=architecture,
            current_config=current_config,
            actions_taken=actions_taken,
        )

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

        self.console.print("[bold yellow]Architecture:[/bold yellow]")
        indent = "    "
        for i, layer in enumerate(layers):
            if hasattr(layer, "layer_type") and layer.layer_type.name == "CONV":
                self.console.print(
                    f"{indent}[bold yellow]Layer {i}:[/bold yellow] {layer.layer_type.name} - OutChannels: {layer.out_channels.name}, Kernel Size: {layer.kernel_size.name}, Stride: {layer.stride.name}, Pool Mode: {layer.pool_mode.name}, Activation: {layer.activation.name}"
                )
            elif hasattr(layer, "layer_type") and layer.layer_type.name == "LINEAR":
                self.console.print(
                    f"{indent}[bold yellow]Layer {i}:[/bold yellow] {layer.layer_type.name} - LinearUnits: {layer.linear_units.name}, Activation: {layer.activation.name}"
                )
            elif hasattr(layer, "layer_type") and layer.layer_type.name == "POOL":
                self.console.print(
                    f"{indent}[bold yellow]Layer {i}:[/bold yellow] {layer.layer_type.name} - Pool Mode: {layer.pool_mode.name}, Kernel Size: {layer.kernel_size.name}, Stride: {layer.stride.name} , Pool Mode: {layer.pool_mode.name}, Activation: {layer.activation.name}"
                )


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
