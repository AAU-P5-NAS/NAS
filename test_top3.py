import torch
from rich.console import Console

from src.environment.train import Trainer

from torch import nn
from src.environment.metrics import Evaluator
from src.utils.data_importer.dataset import DatasetOption
from src.utils.data_importer.importer import DataImporter

from torch.optim import Adam

console = Console()

def get_ws_1():
    return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64 * 8 * 8, 10),
        )
def get_ws_2():
    return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64 * 8 * 8, 10),
        )

def get_ws_3():
    return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64 * 8 * 8, 10),
        )

def main():
    torch.manual_seed(42)
    model = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128 * 32 * 32, 10),
        )
    #[
        # Layer 0: CONV - OutChannels: CH_64, Kernel Size: KS_3, Stride: S_1, Pool Mode: NONE, Activation: RELU
        # Layer 1: CONV - OutChannels: CH_64, Kernel Size: KS_3, Stride: S_1, Pool Mode: NONE, Activation: RELU
        # Layer 2: CONV - OutChannels: CH_64, Kernel Size: KS_3, Stride: S_2, Pool Mode: NONE, Activation: RELU
        # Layer 3: CONV - OutChannels: CH_128, Kernel Size: KS_3, Stride: S_1, Pool Mode: NONE, Activation: RELU
        # Layer 4: CONV - OutChannels: CH_64, Kernel Size: KS_5, Stride: S_1, Pool Mode: NONE, Activation: RELU
        # Layer 5: CONV - OutChannels: CH_128, Kernel Size: KS_3, Stride: S_1, Pool Mode: NONE, Activation: RELU
        # Layer 6: CONV - OutChannels: CH_64, Kernel Size: KS_5, Stride: S_2, Pool Mode: NONE, Activation: RELU
        
        # Layer 0: CONV - OutChannels: CH_64, Kernel Size: KS_3, Stride: S_1, Pool Mode: NONE, Activation: RELU
        # Layer 1: CONV - OutChannels: CH_64, Kernel Size: KS_3, Stride: S_1, Pool Mode: NONE, Activation: RELU
        # Layer 2: CONV - OutChannels: CH_64, Kernel Size: KS_3, Stride: S_2, Pool Mode: NONE, Activation: RELU
        # Layer 3: CONV - OutChannels: CH_128, Kernel Size: KS_3, Stride: S_1, Pool Mode: NONE, Activation: RELU
        # Layer 4: CONV - OutChannels: CH_128, Kernel Size: KS_5, Stride: S_1, Pool Mode: NONE, Activation: RELU
        # Layer 5: CONV - OutChannels: CH_128, Kernel Size: KS_5, Stride: S_1, Pool Mode: NONE, Activation: RELU
        # Layer 6: CONV - OutChannels: CH_64, Kernel Size: KS_3, Stride: S_2, Pool Mode: NONE, Activation: RELU
        
        # Layer 0: CONV - OutChannels: CH_64, Kernel Size: KS_3, Stride: S_1, Pool Mode: NONE, Activation: RELU
        # Layer 1: CONV - OutChannels: CH_64, Kernel Size: KS_3, Stride: S_1, Pool Mode: NONE, Activation: RELU
        # Layer 2: CONV - OutChannels: CH_64, Kernel Size: KS_5, Stride: S_2, Pool Mode: NONE, Activation: RELU
        # Layer 3: CONV - OutChannels: CH_128, Kernel Size: KS_3, Stride: S_1, Pool Mode: NONE, Activation: RELU
        # Layer 4: CONV - OutChannels: CH_64, Kernel Size: KS_5, Stride: S_1, Pool Mode: NONE, Activation: RELU
        # Layer 5: CONV - OutChannels: CH_128, Kernel Size: KS_5, Stride: S_1, Pool Mode: NONE, Activation: RELU
        # Layer 6: CONV - OutChannels: CH_64, Kernel Size: KS_3, Stride: S_2, Pool Mode: NONE, Activation: RELU   
    #]
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
    print(f"training tchebycheff rank #1")
    for epoch in range(75):
        print("Epoch:", epoch + 1)
        trainer.train(model, optimizer)
        metrics = evaluator.evaluate(model)
        console.print(
            f"[bold green]Metrics: Accuracy: {metrics.accuracy:.4f}, FLOPS: {metrics.flops:.2f}[/bold green]"
        )


if __name__ == "__main__":
    main()
