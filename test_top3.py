import traceback
import optuna
import torch
from typing import Dict, Any, Optional
from rich.console import Console

from src.environment.reward.reward import Weights
from src.environment.reward.weighted_sum import WeightedSumRS
from src.environment.train import Trainer

from torch import nn
from src.environment.metrics import Evaluator
from src.utils.data_importer.dataset import DatasetOption
from src.utils.hyperparameters import HyperparameterSearchSpace
from src.utils.data_importer.importer import DataImporter

from src.environment.environment import Trainer
from torch.optim import AdamW
from src.environment.metrics import Evaluator
from src.utils.data_importer.importer import DataImporter

console = Console()


def main():
    model = [None, None, None]
    model[0] = nn.Sequential(
        nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(64),
    )
    model[1] = nn.Sequential(
        nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
    )
    model[2] = nn.Sequential(
        nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
    )

    importer = DataImporter(dataset_option=DatasetOption.CIFAR_10)
    trainer = Trainer(
        dataloaders=importer.get_dataloaders(batch_size=64),
        loss_function=torch.nn.CrossEntropyLoss()
    )
    evaluator = Evaluator(
        num_classes=10,
        dataloaders=importer.get_dataloaders(batch_size=64, shuffle=False),
        dimensions=importer.get_dimensions(),
        device=torch.device("cuda"),
        loss_function=torch.nn.CrossEntropyLoss(),
    )
    optimizer = AdamW(model.parameters(), 0.00132)
    for i in range(3):
        print(f"training arch {i+1}")
        for epoch in range(75):
            trainer.train(model[i], optimizer)
        metrics[i] = evaluator.evaluate(model[i])
        console.print(
            f"[bold green]Metrics for arch {i+1}: Accuracy: {metrics.accuracy:.4f}, FLOPS: {metrics.flops:.2f}[/bold green]"
        )
       
if __name__ == "__main__":
    main()
