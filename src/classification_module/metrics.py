import torch
from pydantic import BaseModel
from torch import Tensor
from torchmetrics import Accuracy, Precision, Recall, F1Score
from typing import Literal, List, Optional, cast


class Metrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    flops: float
    runtime: float
    test_loss: float
    architecture_size: int


class Metrics_modules(BaseModel):
    accuracy: Accuracy
    precision: Precision
    recall: Recall
    f1_score: F1Score


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


class MetrcicsEvaluator:
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        num_classes: int = 26,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    ):
        task: str = "multiclass"
        self.device = device

        self.modules: Metrics_modules = Metrics_modules(
            accuracy=cast(
                Accuracy, Accuracy(task=task, average=average, num_classes=num_classes).to(device)
            ),
            precision=cast(
                Precision, Precision(task=task, average=average, num_classes=num_classes).to(device)
            ),
            recall=cast(
                Recall, Recall(task=task, average=average, num_classes=num_classes).to(device)
            ),
            f1_score=cast(
                F1Score, F1Score(task=task, average=average, num_classes=num_classes).to(device)
            ),
        )

        self.metrics: Metrics = Metrics(
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            test_loss=0.0,
            flops=0.0,
            runtime=0.0,
            architecture_size=0,
        )

    def update_predictoins_and_targets(self, predictions: Tensor, targets: Tensor):
        predictions = predictions.to(self.device)
        targets = targets.to(self.device)
        for module in self.modules.model_dump().values():
            module.update(predictions, targets)

    def compute_metrics(
        self,
        metrics: List[Metric_literal] = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "test_loss",
            "flops",
            "runtime",
            "architecture_size",
        ],
    ):
        for metric_name, module in self.modules.model_dump().items():
            value = module.compute().item()
            self.metrics.__setattr__(metric_name, value)
        return self.metrics

    def compute_flops(self, model, input_size):
        # Placeholder for FLOPs calculation
        pass

    def compute_runtime(self, model, input_data, iterations=100):
        # Placeholder for runtime calculation
        pass

    def compute_architecture_size(self, model):
        return sum(p.numel() for p in model.parameters())

    def calculate_all_metrics(self, model, predictions, targets):
        pass
