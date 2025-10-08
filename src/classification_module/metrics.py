import torch
import time
from pydantic import BaseModel
from torch import Tensor, nn
from torchmetrics import Accuracy, Precision, Recall, F1Score
from typing import Literal, List, Optional, Callable, Union, cast
from fvcore.nn import FlopCountAnalysis
from src.data_module.importer import IMG_DEFAULT_SIZE, NUM_CLASSES


class InvalidMetricError(Exception):
    pass


class Metrics(BaseModel):
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    flops: Optional[int] = None
    runtime: Optional[float] = None
    test_loss: Optional[float] = None
    architecture_size: Optional[int] = None
    training_time: Optional[float] = None


Metric_literal = Literal[
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "test_loss",
    "flops",
    "runtime",
    "architecture_size",
]

Metric_literal_no_test_loss = Literal[
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "flops",
    "runtime",
    "architecture_size",
]


class MetrcicsEvaluator:
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    ):
        task: str = "multiclass"
        self.device = device
        self.image_dimensions = (1, *IMG_DEFAULT_SIZE)

        self.accuracy: Accuracy = cast(
            Accuracy, Accuracy(task=task, average=average, num_classes=NUM_CLASSES).to(device)
        )
        self.precision: Precision = cast(
            Precision, Precision(task=task, average=average, num_classes=NUM_CLASSES).to(device)
        )
        self.recall: Recall = cast(
            Recall, Recall(task=task, average=average, num_classes=NUM_CLASSES).to(device)
        )
        self.f1_score: F1Score = cast(
            F1Score, F1Score(task=task, average=average, num_classes=NUM_CLASSES).to(device)
        )

    def calculate_metrics(
        self,
        model: nn.Module,
        predictions: Tensor,
        targets: Tensor,
        metrics: List[Metric_literal_no_test_loss] = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "flops",
            "runtime",
            "architecture_size",
        ],
    ) -> Metrics:
        computed_metrics: Metrics = Metrics()

        for metric in metrics:
            method_name = f"compute_{metric}"
            try:
                method: Callable[..., Union[int, float]] = self.__getattribute__(method_name)
            except AttributeError:
                raise InvalidMetricError(f"Metric '{metric}' is not supported.")

            value: Union[int, float]
            if metric in ["flops", "runtime", "architecture_size"]:
                value = method(model)
            elif metric == "test_loss":
                pass
            else:
                value = method(predictions, targets)

            computed_metrics.__setattr__(metric, value)

        return computed_metrics

    def compute_flops(self, model: nn.Module, batch_size: int = 1) -> int:
        dummy_input: Tensor = torch.randn(batch_size, *self.image_dimensions).to(self.device)
        total_flops: int = FlopCountAnalysis(model, dummy_input).total()
        return total_flops

    def compute_runtime(self, model: nn.Module, iterations: int = 50, batch_size: int = 1) -> float:
        warmup_iterations: int = 10

        model.eval()
        dummy_input: Tensor = torch.randn(batch_size, *self.image_dimensions).to(self.device)

        # Warm-up cold GPU
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(dummy_input)

        if iterations <= 0:
            raise ValueError("iterations must be positive")

        # Measure runtime
        if self.device.type == "cuda":
            torch.cuda.synchronize()  # Ensure GPU is ready (wait until prior scheduled tasks are done)
            start_time: float = time.perf_counter()
            with torch.no_grad():
                for _ in range(iterations):
                    _ = model(dummy_input)
            torch.cuda.synchronize()  # Wait for GPU to finish all itterations
        else:
            start_time: float = time.perf_counter()
            with torch.no_grad():
                for _ in range(iterations):
                    _ = model(dummy_input)

        end_time: float = time.time()
        avg_runtime: float = (end_time - start_time) / iterations
        return avg_runtime

    def compute_architecture_size(self, model) -> int:
        return sum(p.numel() for p in model.parameters())

    def compute_accuracy(self, predictions: Tensor, targets: Tensor) -> float:
        return self.accuracy(predictions, targets).item()

    def compute_precision(self, predictions: Tensor, targets: Tensor) -> float:
        return self.precision(predictions, targets).item()

    def compute_recall(self, predictions: Tensor, targets: Tensor) -> float:
        return self.recall(predictions, targets).item()

    def compute_f1_score(self, predictions: Tensor, targets: Tensor) -> float:
        return self.f1_score(predictions, targets).item()
