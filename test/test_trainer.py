import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.classification_module.train import train

X = torch.tensor([1, 1])
y = torch.tensor([1])
test_dataset = TensorDataset(X, y)

X_empty = torch.empty(0)
y_empty = torch.empty(0)
empty_dataset = TensorDataset(X_empty, y_empty)


def test_empty_dataloader():
    dataloader = DataLoader(empty_dataset, batch_size=2)
    model = torch.nn.Linear(1, 1)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters())
    
    with pytest.raises(ValueError):
        train(
            dataloader,
            model,
            loss_function,
            optimizer,
        )


#def test_():
    dataloader = DataLoader(test_dataset, batch_size=2)
    model = torch.nn.Linear(1, 1)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters())
    
    with pytest.raises(ValueError):
        train(
            dataloader,
            model,
            loss_function,
            optimizer,
        )
