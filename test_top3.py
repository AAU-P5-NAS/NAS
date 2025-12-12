import torch
from rich.console import Console

from src.environment.train import Trainer

from torch import nn
from src.environment.metrics import Evaluator
from src.utils.data_importer.dataset import DatasetOption
from src.utils.data_importer.importer import DataImporter

from torch.optim import AdamW

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
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(64, 10),
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
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(64, 10),
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
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(128, 10),
    )
    metrics = [None, None, None]
    importer = DataImporter(dataset_option=DatasetOption.CIFAR_10)
    trainer = Trainer(
        dataloaders=importer.get_dataloaders(batch_size=64),
        loss_function=torch.nn.CrossEntropyLoss(),
    )
    evaluator = Evaluator(
        num_classes=10,
        dataloaders=importer.get_dataloaders(batch_size=64, shuffle=False),
        dimensions=importer.get_dimensions(),
        device=torch.device("cuda"),
        loss_function=torch.nn.CrossEntropyLoss(),
    )
    optimizer = AdamW(model[0].parameters(), 0.00132)
    for i in range(3):
        print(f"training arch {i + 1}")
        for epoch in range(75):
            print("Epoch:", epoch + 1)
            trainer.train(model[i], optimizer)
        metrics[i] = evaluator.evaluate(model[i])
        console.print(
            f"[bold green]Metrics for arch {i + 1}: Accuracy: {metrics[i].accuracy:.4f}, FLOPS: {metrics[i].flops:.2f}[/bold green]"
        )


if __name__ == "__main__":
    main()
