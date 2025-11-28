from typing import Optional
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from typing import Tuple
from threading import Event, Timer


class Trainer:
    def __init__(
        self,
        dataloaders: Tuple[DataLoader, DataLoader],
        loss_function: _Loss,
    ):
        self.train_loader, self.test_loader = dataloaders
        self.device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.loss_function: _Loss = loss_function.to(self.device)

    def train(self, model: nn.Module, optimizer: Optimizer, max_training_time: Optional[int] = 300):
        model = model.to(self.device)

        # Set up max training timer
        stop_event = Event()
        timer = None
        if max_training_time is not None:
            timer = Timer(max_training_time, stop_event.set)
            timer.daemon = True
            timer.start()

        stopped_by_timeout = False

        # Train the model
        model.train()
        for X, Y in self.train_loader:
            # Stop training if it takes too long
            if stop_event.is_set():
                stopped_by_timeout = True
                break

            # Ensure input tensors are float32 when feeding the model. Some
            # test fixtures or dataloaders may yield float64 which causes
            # mismatches with model parameters (float32) on some devices.
            X = X.to(self.device, non_blocking=True).float()
            Y = Y.to(self.device, non_blocking=True)

            # Compute prediction error
            predictions = model(X)
            loss = self.loss_function(predictions, Y)
            # Defensive checks: detect NaN/Inf loss and stop early so training of
            # this candidate architecture doesn't silently corrupt downstream RL.
            if not torch.isfinite(loss):
                try:
                    print(
                        "[warning] Trainer detected non-finite loss (NaN or Inf). Aborting this training run."
                    )
                except Exception:
                    pass
                stopped_by_timeout = True
                break
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Check for NaNs in gradients before update
            grads_nan = False
            for p in model.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        grads_nan = True
                        break
            if grads_nan:
                try:
                    print("[warning] Trainer detected NaN/Inf in gradients — aborting update.")
                except Exception:
                    pass
                stopped_by_timeout = True
                break

            # Clip large gradients to prevent explosion
            try:
                from torch.nn.utils import clip_grad_norm_

                clip_grad_norm_(model.parameters(), max_norm=1.0)
            except Exception:
                # If clipping fails, ignore to avoid blocking minimal environment runs
                pass
            optimizer.step()
        # Clear timer
        if timer is not None:
            timer.cancel()

        # Return if stopped prematurely
        return stopped_by_timeout


""" 
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
 """
