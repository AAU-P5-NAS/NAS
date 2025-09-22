import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.modules.loss import _Loss
from typing import cast
from torch import nn


def test(dataloader: DataLoader[tuple[torch.Tensor, ...]], model: nn.Module, loss_function: _Loss):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss: float = 0
    correct: int = 0
    cardinality: int = len(cast(TensorDataset, dataloader.dataset))
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            predictions = model(X)
            test_loss += loss_function(predictions, y).item()
            correct += (predictions.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= len(dataloader)
    accuracy = correct / cardinality

    return (accuracy, test_loss)
