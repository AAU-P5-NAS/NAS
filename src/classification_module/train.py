import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from typing import Tuple, List
from src.classification_module.metrics import MetrcicsEvaluator
from src.classification_module.metrics import Metric_literal, Metrics


class Trainer:
    def __init__(
        self,
        dataloaders: Tuple[DataLoader, DataLoader],
        model: nn.Module,
        loss_function: _Loss,
        optimizer: Optimizer,
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
        self.loss_function: _Loss = loss_function
        self.optimizer: Optimizer = optimizer
        self.evaluator: MetrcicsEvaluator = MetrcicsEvaluator()
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chosen_metrics: List[Metric_literal] = chosen_metrics

    def train(self):
        self.model.train()
        for X, y in self.train_loader:
            X = X.to(self.device)
            y = y.to(self.device)

            # Compute prediction error
            predictions = self.model(X)
            loss = self.loss_function(predictions, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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
                all_preds.append(predictions.cpu())
                all_labels.append(y.cpu())

        test_loss /= len(self.test_loader)

        all_preds_flattened: Tensor = torch.cat(all_preds)
        all_labels_flattened: Tensor = torch.cat(all_labels)

        metrics: Metrics = self.evaluator.calculate_metrics(
            self.model, all_preds_flattened, all_labels_flattened
        )
        if "test_loss" in self.chosen_metrics:
            metrics.__setattr__("test_loss", test_loss)

        return metrics
