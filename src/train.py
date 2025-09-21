from utils.ffn_utils import make_ffn, export_ffn_to_onnx
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from typing import cast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# how too build a FFN
ffn = make_ffn([(28 * 28, 100, "relu"), (100, 10, "softmax")])

# How too export this to a ONNX file
export_ffn_to_onnx(ffn, input_size=28 * 28, filename="fnn.onnx")

# ----- Training network -----


def train(dataloader: DataLoader, model: nn.Module, loss_function: _Loss, optimizer: Optimizer):
    model.train()
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        # Compute prediction error
        predictions = model(X)
        loss = loss_function(predictions, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(dataloader: DataLoader, model: nn.Module, loss_function: _Loss):
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
