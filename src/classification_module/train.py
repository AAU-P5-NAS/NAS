import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from typing import Tuple, List
from src.classification_module.metrics import MetrcicsEvaluator
from src.classification_module.metrics import Metric_literal, Metrics
import traceback


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
        device: torch.device | None = None,
    ):
        self.train_loader, self.test_loader = dataloaders
        self.model: nn.Module = model
        self.evaluator: MetrcicsEvaluator = MetrcicsEvaluator()
        self.device: torch.device | None = device
        self.model.to(self.device)
        self.loss_function: _Loss = loss_function.to(self.device)
        self.optimizer: Optimizer = optimizer
        self.chosen_metrics: List[Metric_literal] = chosen_metrics

    def train(self):
        print("DEBUG model device after to(device):", next(self.model.parameters()).device)
        self.model.train()
        print("I REACHED THIS HIDDEN PLACE")
        for X, y in self.train_loader:
            X = X.to(self.device)
            y = y.to(self.device)

            predictions = self.model(X)
            # DEBUG WRAPPER: catch forward errors and record intermediate device info
            try:
                predictions = self.model(X)
            except Exception:
                traceback.print_exc()

                # show named param/buffer devices again
                print("PARAM devices:", {n: p.device for n, p in self.model.named_parameters()})
                print("BUFFER devices:", {n: b.device for n, b in self.model.named_buffers()})

                # register hooks to print output device(s) per module on a single forward run
                hooks = []

                def mk_hook(name):
                    def hook(module, inp, out):
                        def _dev(o):
                            if torch.is_tensor(o):
                                return o.device
                            if isinstance(o, (list, tuple)):
                                for x in o:
                                    d = _dev(x)
                                    if d is not None:
                                        return d
                            return None

                        print("HOOK:", name, "output_device:", _dev(out))

                    return hook

                for nm, m in self.model.named_modules():
                    hooks.append(m.register_forward_hook(mk_hook(nm)))

                # run one forward to trigger hooks (wrap in try to still print any exception)
                try:
                    _ = self.model(X)
                except Exception:
                    traceback.print_exc()

                # remove hooks
                for h in hooks:
                    h.remove()

                # quick grep hint for common mistakes
                print("Search for cpu/numpy/plain-tensor creation in model code:")
                print(
                    '  grep -nR -E "torch\\.(tensor|zeros|ones|arange)|\\.cpu\\(|\\.numpy\\(" src/'
                )
                # re-raise so environment can see the failure if you want
                raise
            print("DEBUG model param devices:", {p.device for p in self.model.parameters()})
            print("DEBUG input device:", getattr(X, "device", None))
            print("DEBUG target device:", getattr(y, "device", None))
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

                # DEBUG: same diagnostics in test
                param_devs = {p.device for p in self.model.parameters()}
                print("DEBUG model param devices:", param_devs)
                print("DEBUG input device:", getattr(X, "device", None))
                # End DEBUG

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
