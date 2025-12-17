import torch
from rich.console import Console
from src.environment.train import Trainer
from src.environment.metrics import Evaluator
from src.utils.data_importer.dataset import DatasetOption
from src.utils.data_importer.importer import DataImporter
from torch.optim import Adam
from torch import nn


"""
Architecture Rank 1:
Metrics: accuracy=None precision=None recall=None f1_score=None flops=None runtime=None test_loss=None architecture_size=None training_time=None synflow=7300077.5 jacov=7.136654858186375e-07 snip=2165.454345703125 complexity=2658122.0

Layer 0: CONV - OutChannels: CH_128, Kernel Size: KS_5, Stride: S_1, Pool Mode: NONE, Activation: NONE

Layer 1: CONV - OutChannels: CH_128, Kernel Size: KS_5, Stride: S_1, Pool Mode: NONE, Activation: RELU

Layer 2: CONV - OutChannels: CH_128, Kernel Size: KS_5, Stride: S_1, Pool Mode: NONE, Activation: NONE

Layer 3: CONV - OutChannels: CH_128, Kernel Size: KS_5, Stride: S_1, Pool Mode: NONE, Activation: RELU

Layer 4: CONV - OutChannels: CH_128, Kernel Size: KS_3, Stride: S_1, Pool Mode: NONE, Activation: NONE

Layer 5: CONV - OutChannels: CH_128, Kernel Size: KS_5, Stride: S_1, Pool Mode: NONE, Activation: NONE

Layer 6: CONV - OutChannels: CH_64, Kernel Size: KS_5, Stride: S_1, Pool Mode: NONE, Activation: NONE

================================================================================

Architecture Rank 2:
Metrics: accuracy=None precision=None recall=None f1_score=None flops=None runtime=None test_loss=None architecture_size=None training_time=None synflow=7804747.0 jacov=3.8788422784818977e-07 snip=2993.680908203125 complexity=3518474.0

Layer 0: CONV - OutChannels: CH_128, Kernel Size: KS_5, Stride: S_1, Pool Mode: NONE, Activation: NONE

Layer 1: CONV - OutChannels: CH_128, Kernel Size: KS_5, Stride: S_1, Pool Mode: NONE, Activation: RELU

Layer 2: CONV - OutChannels: CH_128, Kernel Size: KS_3, Stride: S_1, Pool Mode: NONE, Activation: NONE

Layer 3: CONV - OutChannels: CH_128, Kernel Size: KS_5, Stride: S_1, Pool Mode: NONE, Activation: NONE

Layer 4: CONV - OutChannels: CH_128, Kernel Size: KS_5, Stride: S_1, Pool Mode: NONE, Activation: RELU

Layer 5: CONV - OutChannels: CH_128, Kernel Size: KS_5, Stride: S_1, Pool Mode: NONE, Activation: RELU

Layer 6: CONV - OutChannels: CH_128, Kernel Size: KS_5, Stride: S_1, Pool Mode: NONE, Activation: NONE

================================================================================

Architecture Rank 3:
Metrics: accuracy=None precision=None recall=None f1_score=None flops=None runtime=None test_loss=None architecture_size=None training_time=None synflow=7784161.0 jacov=2.231248430462074e-07 snip=2948.956787109375 complexity=3518474.0

Layer 0: CONV - OutChannels: CH_128, Kernel Size: KS_5, Stride: S_1, Pool Mode: NONE, Activation: RELU

Layer 1: CONV - OutChannels: CH_128, Kernel Size: KS_5, Stride: S_1, Pool Mode: NONE, Activation: RELU

Layer 2: CONV - OutChannels: CH_128, Kernel Size: KS_5, Stride: S_1, Pool Mode: NONE, Activation: RELU

Layer 3: CONV - OutChannels: CH_128, Kernel Size: KS_5, Stride: S_1, Pool Mode: NONE, Activation: NONE

Layer 4: CONV - OutChannels: CH_128, Kernel Size: KS_3, Stride: S_1, Pool Mode: NONE, Activation: NONE

Layer 5: CONV - OutChannels: CH_128, Kernel Size: KS_5, Stride: S_1, Pool Mode: NONE, Activation: RELU

Layer 6: CONV - OutChannels: CH_128, Kernel Size: KS_5, Stride: S_1, Pool Mode: NONE, Activation: NONE

================================================================================
"""

console = Console()


def main():
    torch.manual_seed(42)
    model_1 = nn.Sequential(
        # Layer 0
        nn.Conv2d(3, 128, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(128),
        # No activation
        # Layer 1
        nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        # Layer 2
        nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(128),
        # No activation
        # Layer 3
        nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        # Layer 4
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        # No activation
        # Layer 5
        nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(128),
        # No activation
        # Layer 6
        nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(64),
        # No activation
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(64 * 32 * 32, 10),
    )

    model_2 = nn.Sequential(
        # Layer 0
        nn.Conv2d(3, 128, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(128),
        # No activation
        # Layer 1
        nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        # Layer 2
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        # No activation
        # Layer 3
        nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(128),
        # No activation
        # Layer 4
        nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        # Layer 5
        nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        # Layer 6
        nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(128),
        # No activation
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(128 * 32 * 32, 10),
    )

    model_3 = nn.Sequential(
        # Layer 0
        nn.Conv2d(3, 128, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        # Layer 1
        nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        # Layer 2
        nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        # Layer 3
        nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(128),
        # No activation
        # Layer 4
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        # No activation
        # Layer 5
        nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        # Layer 6
        nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(128),
        # No activation
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(128 * 32 * 32, 10),
    )

    best_arch = [
        (model_1, "model_1"),
        (model_2, "model_2"),
        (model_3, "model_3"),
    ]

    validated_metrics: dict[str, list] = {}
    for model, name in best_arch:
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
        optimizer = Adam(model.parameters(), 0.00132)

        for epoch in range(100):
            print("Epoch:", epoch + 1)
            trainer.train(model, optimizer)
            metrics = evaluator.evaluate(model)
            console.print(
                f"[bold green]{name}, Metrics: Accuracy: {metrics.accuracy:.4f}, FLOPS: {metrics.flops:.2f}[/bold green]"
            )

            if validated_metrics.get(name) is None or validated_metrics[name][0] < metrics.accuracy:
                validated_metrics[name] = [metrics.accuracy, metrics.flops]

    console.print("\n[bold yellow]Final validated metrics:[/bold yellow]")
    for name, metric in validated_metrics.items():
        console.print(
            f"[bold green]{name}, Metrics: Accuracy: {metric[0]:.4f}, FLOPS: {metric[1]:.2f}[/bold green]"
        )


if __name__ == "__main__":
    main()
