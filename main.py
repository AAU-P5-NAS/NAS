import torch
import time
from src.data_module.importer import DataImporter
from src.classification_module.train import Trainer
from src.classification_module.metrics import Metrics
from rich.console import Console

from src.utils.CNNBuilder import (
    CNNBuilder,
    NetworkConfig,
    LayerConfig,
    LayerType,
    OutChannels,
    KernelSize,
    LinearUnits,
    ActivationFunction,
    PoolMode,
)


def main():
    # 1. Load data
    importer = DataImporter()
    train_loader, test_loader = importer.get_as_cnn(batch_size=512)

    console = Console()

    # 2. Define CNN configuration using CNNActionSpace
    config = NetworkConfig(
        layers=[
            LayerConfig(
                layer_type=LayerType.CONV,
                out_channels=OutChannels.CH_16,
                kernel_size=KernelSize.KS_3,
                activation=ActivationFunction.RELU,
            ),
            LayerConfig(
                layer_type=LayerType.POOL,
                pool_mode=PoolMode.MAX,
                kernel_size=KernelSize.KS_1,
            ),
            LayerConfig(
                layer_type=LayerType.CONV,
                out_channels=OutChannels.CH_32,
                kernel_size=KernelSize.KS_3,
                activation=ActivationFunction.RELU,
            ),
            LayerConfig(
                layer_type=LayerType.POOL,
                pool_mode=PoolMode.MAX,
                kernel_size=KernelSize.KS_1,
            ),
            LayerConfig(
                layer_type=LayerType.LINEAR,
                linear_units=LinearUnits.LU_64,
                activation=ActivationFunction.RELU,
            ),
        ]
    )

    cnn = CNNBuilder(config)
    model = cnn.build()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    trainer = Trainer(dataloader, model, loss_fn, optimizer)

    # 5. Training loop
    num_epochs = 1
    start_time = time.time()
    for epoch in range(num_epochs):
        with console.status(f"[bold blue]Training CNN, epoch {epoch + 1}/{num_epochs}"):
            trainer.train()

    end_time = time.time()
    training_time = end_time - start_time

    metrics: Metrics = trainer.test()
    metrics.training_time = training_time
    console.print(f"[bold green]Epoch {epoch + 1} complete![/bold green]")
    console.print(f"[bold yellow]Test Metrics:[/bold yellow] {metrics.model_dump()}")

    console.print("[bold blue]CNN training complete![/bold blue]")


if __name__ == "__main__":
    main()
