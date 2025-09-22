import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer


def train(
    dataloader: DataLoader[tuple[torch.Tensor, ...]],
    model: nn.Module,
    loss_function: _Loss,
    optimizer: Optimizer,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for X, y in dataloader:
        print("X batch shape:", X.shape)
        X = X.to(device)
        y = y.to(device)

        # Compute prediction error
        predictions = model(X)
        loss = loss_function(predictions, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
