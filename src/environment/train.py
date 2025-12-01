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

    def train(self,model: nn.Module, optimizer: Optimizer, max_training_time: Optional[int] = 300):
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

            X = X.to(self.device, non_blocking=True)
            Y = Y.to(self.device, non_blocking=True)

            # Compute prediction error
            predictions = model(X)
            loss = self.loss_function(predictions, Y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Clear timer
        if timer is not None:
            timer.cancel()

        # Return if stopped prematurely
        return stopped_by_timeout