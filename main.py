import torch
from src.data_module.importer import DataImporter
from src.classification_module.train import train
from rich.console import Console
from src.classification_module.test import test
from src.utils.CNNBuilder import CNNBuilder  # your CNNBuilder class


def main():
    # 1. Load data
    importer = DataImporter()
    train_loader = importer.get_as_cnn(batch_size=512)  # returns just the DataLoader
    test_loader = importer.get_as_cnn(batch_size=512)  # for evaluation

    console = Console()

    # 2. Define CNN configuration
    config = [
        {"layer_type": "conv", "out_channels": 16, "kernel_size": 3, "activation": "relu"},
        {"layer_type": "pool", "pool_mode": "max", "kernel_size": 2},
        {"layer_type": "conv", "out_channels": 32, "kernel_size": 3, "activation": "relu"},
        {"layer_type": "pool", "pool_mode": "max", "kernel_size": 2},
        {"layer_type": "linear", "linear_units": 64, "activation": "relu"},
    ]

    # 3. Build CNN model
    cnn = CNNBuilder(config, in_channels=1, input_size=(28, 28), num_classes=26)
    model = cnn.build()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 4. Loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 5. Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        with console.status(f"[bold blue]Training CNN, epoch {epoch + 1}/{num_epochs}"):
            train(train_loader, model, loss_fn, optimizer)

        accuracy, test_loss = test(test_loader, model, loss_fn)
        console.print(f"[bold green]Epoch {epoch + 1} complete![/bold green]")
        console.print(
            f"[bold yellow]Test Accuracy: {accuracy:.4f}, Test Loss: {test_loss:.4f}[/bold yellow]"
        )

    console.print("[bold blue]CNN training complete![/bold blue]")

    # 6. Optional: export to ONNX
    onnx_path = cnn.export_to_onnx(input_size=(1, 28, 28), filename="cnn_model.onnx")
    console.print(f"[bold magenta]Model exported to {onnx_path}[/bold magenta]")


if __name__ == "__main__":
    main()


# def fnn():
#     importer = DataImporter()
#     console = Console()
#     model = make_ffn([(28 * 28, 16, "relu"), (16, 26, None)])
#     network = Network(model)
#     train_dataloader, test_dataloader = importer.get_as_ffnn(batch_size=1, test_split=0.1)
#     # for X, y in train_dataloader:
#     #     print("y ", y.shape)

#     for epoch in range(25):
#         with console.status(f"[bold blue] Training model, epoch {epoch} / 25"):
#             train(
#                 train_dataloader,
#                 network,
#                 torch.nn.CrossEntropyLoss(),
#                 torch.optim.SGD(network.parameters(), lr=0.01),
#             )
#         console.print(f"[bold green]Epoch {epoch + 1} ✔[/bold green]")
#         accuracy, test_loss = test(test_dataloader, network, torch.nn.CrossEntropyLoss())
#         console.print(
#             f"[bold yellow]Test accuracy: {accuracy:.4f}, Test loss: {test_loss:.4f}[/bold yellow]"
#         )

#     console.print("[bold blue]Training complete![/bold blue]")
