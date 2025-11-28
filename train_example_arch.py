import time
from src.environment.metrics import Evaluator
from src.utils.data_importer.dataset import DatasetOption
from src.environment.train import Trainer
from src.utils.data_importer.importer import DataImporter

import torch
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from rich.console import Console

console = Console()


def main():
    importer = DataImporter(dataset_option=DatasetOption.CIFAR_10)
    dataloaders = importer.get_dataloaders(batch_size=32, shuffle=True)
    number_of_classes = 26
    print("is cuda available:", torch.cuda.is_available())
    number_of_classes = 10
    evaluator = Evaluator(num_classes=number_of_classes, 
                          dataloaders=dataloaders, 
                          dimensions=importer.get_dimensions(), 
                          device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                          loss_function=CrossEntropyLoss())

    model = nn.Sequential(
        # Block 1
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.25),
        # Block 2
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.25),
        # Block 3
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.25),
        # Fully connected
        nn.Flatten(),
        nn.Linear(256 * 4 * 4, 512),  # Adjusted input size after removing Block 4
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, number_of_classes),
    )

    for layer in model:
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)  # Xavier/Glorot uniform initialization
            if layer.bias is not None:
                layer.bias.data.fill_(0.0)  # Zero bias

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(
        dataloaders=dataloaders,
        loss_function=CrossEntropyLoss().to(device),
    )
    num_epochs = 10
    start_time = time.time()

    for epoch in range(num_epochs):
        progress = (epoch + 1) / num_epochs * 100
        with console.status(
            f"[bold blue]Training model on epoch {epoch}/{num_epochs}: Progress {int(progress)}%[/bold blue]"
        ):
            trainer.train(model, optimizer)

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    metrics = evaluator.evaluate(model)
    metrics.runtime = training_time
    metrics.training_time = training_time
    console.print("Metrics:", metrics)
    return metrics

    # should import evaluator if u want to test something.

""" 
    metrics.runtime = training_time
    metrics.training_time = training_time
    console.print("Metrics:", metrics)
    return metrics
 """

if __name__ == "__main__":
    main()
