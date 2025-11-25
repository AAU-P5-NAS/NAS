from typing import Optional
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from typing import Tuple, List
from src.environment.metrics import Metric_literal, Metrics, MetrcicsEvaluator
from threading import Event, Timer


class Trainer:
    def __init__(
        self,
        dataloaders: Tuple[DataLoader, DataLoader],
        model: nn.Module,
        loss_function: _Loss,
        optimizer: Optimizer,
        num_classes: int,
        dimensions: tuple[int, int, int],
        chosen_metrics: List[Metric_literal] = [
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
        self.train_loader, self.test_loader = dataloaders
        self.model: nn.Module = model
        self.device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # IMPORTANT: Pass device to evaluator so metrics are on same device as model
        self.evaluator: MetrcicsEvaluator = MetrcicsEvaluator(
            device=self.device, num_classes=num_classes, dimensions=dimensions
        )
        self.model.to(self.device)
        self.loss_function: _Loss = loss_function.to(self.device)
        self.optimizer: Optimizer = optimizer
        self.chosen_metrics: List[Metric_literal] = chosen_metrics

    def train(self, max_training_time: Optional[int] = 300):
        # Set up max training timer
        stop_event = Event()
        timer = None
        if max_training_time is not None:
            timer = Timer(max_training_time, stop_event.set)
            timer.daemon = True
            timer.start()

        stopped_by_timeout = False

        # Train the model
        self.model.train()
        for X, Y in self.train_loader:
            # Stop training if it takes too long
            if stop_event.is_set():
                stopped_by_timeout = True
                break

            X = X.to(self.device, non_blocking=True)
            Y = Y.to(self.device, non_blocking=True)

            # Compute prediction error
            predictions = self.model(X)
            loss = self.loss_function(predictions, Y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # Clear timer
        if timer is not None:
            timer.cancel()

        # Return if stopped prematurely
        return stopped_by_timeout

    def test(self) -> Metrics:
        self.model.eval()
        test_loss: float = 0

        all_preds: list[Tensor] = []
        all_labels: list[Tensor] = []

        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                test_loss += self.loss_function(outputs, y).item()
                predictions = outputs.argmax(1)
                # Keep predictions on same device for metrics calculation
                all_preds.append(predictions)
                all_labels.append(y)

        test_loss /= len(self.test_loader)

        all_preds_flattened: Tensor = torch.cat(all_preds)
        all_labels_flattened: Tensor = torch.cat(all_labels)

        # Predictions and labels are now on the same device as the metrics
        metrics: Metrics = self.evaluator.calculate_metrics(
            self.model,
            all_preds_flattened,
            all_labels_flattened,
            [m for m in self.chosen_metrics if m != "test_loss"],
        )
        if "test_loss" in self.chosen_metrics:
            metrics.__setattr__("test_loss", test_loss)

        return metrics
