import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from typing import cast
from torch.utils.data import TensorDataset
from typing import Tuple

class Trainer:
    def __init__(self, dataloaders: Tuple[DataLoader, DataLoader], model: nn.Module, loss_function: _Loss, optimizer: Optimizer):
        self.train_loader, self.test_loader = dataloaders
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    def test(self):
        self.model.eval()
        test_loss: float = 0
        correct: int = 0
        cardinality: int = len(cast(TensorDataset, self.test_loader.dataset))
        
        with torch.no_grad():
            for X, y in self.test_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                predictions = self.model(X)
                test_loss += self.loss_function(predictions, y).item()
                correct += (predictions.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= len(self.test_loader)
        accuracy = correct / cardinality

        return (accuracy, test_loss)