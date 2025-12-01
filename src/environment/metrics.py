import torch
import time
from pydantic import BaseModel
from torch import Tensor, nn
from torchmetrics import Accuracy, Precision, Recall, F1Score
from typing import Literal, List, Optional, Callable, Tuple, Union, cast
from fvcore.nn import FlopCountAnalysis
from torch.utils.data import DataLoader


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


class TrainingFreeMetrics(Metrics):
    synflow: Optional[float] = None
    jacov: Optional[float] = None
    snip: Optional[float] = None
    complexity: Optional[float] = None

    def normalized(self):
        normalized_metrics = TrainingFreeMetrics()
        for field_name, value in self.model_dump().items():
            if value is not None:
                normalized_value = (value - proxy_baselines[field_name][0]) / (
                    proxy_baselines[field_name][1] - proxy_baselines[field_name][0]
                )
                normalized_metrics.__setattr__(field_name, normalized_value)
        return normalized_metrics


proxy_baselines = {
    "synflow": (1.0, 1e6),
    "jacov": (0.0, 1.0),
    "snip": (0.0, 500.0),
    "complexity": (1e3, 1e7),
}


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


class Evaluator:
    def __init__(
        self,
        num_classes: int,
        dataloaders: Tuple[DataLoader, DataLoader],
        dimensions: tuple[int, int, int],
        loss_function: Callable[[Tensor, Tensor], Tensor],
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    ):
        task: str = "multiclass"
        self.device = device
        self.image_dimensions = dimensions
        self.train_loader, self.test_loader = dataloaders
        self.loss_function = loss_function
        self.accuracy: Accuracy = cast(
            Accuracy, Accuracy(task=task, average=average, num_classes=num_classes).to(device)
        )
        self.precision: Precision = cast(
            Precision, Precision(task=task, average=average, num_classes=num_classes).to(device)
        )
        self.recall: Recall = cast(
            Recall, Recall(task=task, average=average, num_classes=num_classes).to(device)
        )
        self.f1_score: F1Score = cast(
            F1Score, F1Score(task=task, average=average, num_classes=num_classes).to(device)
        )

    def evaluate_by_proxy(
        self,
        model: nn.Module,
    ) -> TrainingFreeMetrics:
        model.to(self.device)
        print("complexity: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
        return TrainingFreeMetrics(
            jacov=self.compute_jacov_proxy(model),
            synflow=self.compute_synflow_proxy(model),
            snip=self.compute_snip_proxy(model),
            complexity=sum(p.numel() for p in model.parameters() if p.requires_grad),
        )

    def evaluate(
        self,
        model: nn.Module,
        training_time: Optional[float] = None,
    ) -> Metrics:
        model = model.to(self.device)
        model.eval()
        test_loss: float = 0.0

        all_preds: List[Tensor] = []
        all_labels: List[Tensor] = []

        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = model(X)
                test_loss += self.loss_function(outputs, y).item()
                predictions = outputs.argmax(1)
                all_preds.append(predictions)
                all_labels.append(y)

        test_loss /= len(self.test_loader)

        all_preds_flattened: Tensor = torch.cat(all_preds)
        all_labels_flattened: Tensor = torch.cat(all_labels)

        computed_metrics: Metrics = self.calculate_metrics(
            model,
            all_preds_flattened,
            all_labels_flattened,
            [m for m in Metric_literal_no_test_loss.__args__],
        )

        computed_metrics.training_time = training_time
        computed_metrics.runtime = training_time

        return computed_metrics

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

        end_time: float = time.perf_counter()
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

    def compute_jacov_proxy(self, model: nn.Module, input_shape=(128, 3, 32, 32)):
        model.eval()
        device = next(model.parameters()).device

        x = torch.randn(input_shape, device=device)

        with torch.no_grad():
            output = model(x)

        output_flat = output.view(output.size(0), -1)
        mean = output_flat.mean(dim=0, keepdim=True)
        centered = output_flat - mean
        cov = (centered.T @ centered) / (output_flat.size(0) - 1)

        # Normalize by output dimension for comparability
        score = torch.norm(cov, p="fro") / output_flat.size(1)

        # print(f"Jacov Score: {score.item():.6f}")
        return score.item()

    def compute_synflow_proxy(self, model: nn.Module, input_shape=(1, 3, 32, 32)):
        model.eval()
        device = next(model.parameters()).device

        # Save original weights on CPU to avoid doubling GPU memory
        original_weights = {name: param.data.cpu().clone() for name, param in model.named_parameters()}

        # Make weights positive in-place (on GPU)
        with torch.no_grad():
            for param in model.parameters():
                param.data.abs_()

        x = torch.ones(input_shape, device=device)
        model.zero_grad()  # clear any existing grads

        # Need grads for synflow
        output = model(x)
        score = output.sum()
        score.backward()

        total = torch.tensor(0.0, device=device)
        for param in model.parameters():
            if param.grad is not None:
                total += torch.sum(torch.abs(param.data * param.grad))

        # Restore original weights from CPU copies
        for name, param in model.named_parameters():
            param.data.copy_(original_weights[name].to(device))

        # Clear grads (important!)
        _clear_grads(model)

        # Delete the CPU backup and free cache
        del original_weights
        if device.type == "cuda":
            torch.cuda.empty_cache()

        num_params = sum(p.numel() for p in model.parameters())
        total = total / num_params
        return total.item()
    
    def compute_snip_proxy(self, model: nn.Module, input_shape=(64, 3, 32, 32)):
        model.eval()
        device = next(model.parameters()).device

        x = torch.randn(input_shape, device=device)
        criterion = nn.CrossEntropyLoss()

        num_classes = 10
        y = torch.randint(0, num_classes, (x.size(0),), device=device)

        model.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        score = torch.tensor(0.0, device=device)
        for param in model.parameters():
            if param.grad is not None:
                score += torch.sum(torch.abs(param.grad * param.data))

        # Normalize
        num_params = sum(p.numel() for p in model.parameters())
        score = score / num_params

        # Clear grads (release GPU memory)
        _clear_grads(model)
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return score.item()


def _clear_grads(model: nn.Module):
    # Prefer p.grad = None because it frees the storage instead of filling with zeros.
    for p in model.parameters():
        p.grad = None