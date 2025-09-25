import torch
from src.data_module.importer import DataImporter
from src.classification_module.train import train
from src.classification_module.torch_network import Network
from rich.console import Console
from src.classification_module.test import test
from src.utils.ffn_utils import make_ffn


def main():
    importer = DataImporter()
    console = Console()
    model = make_ffn([(28 * 28, 32, "relu"), (32, 26, None)])
    network = Network(model)
    train_dataloader, test_dataloader = importer.get_as_ffnn(batch_size=1, test_split=0.1)
    for epoch in range(25):
        with console.status(f"[bold blue] Training model, epoch {epoch} / 25"):
            train(
                train_dataloader,
                network,
                torch.nn.CrossEntropyLoss(),
                torch.optim.SGD(network.parameters(), lr=0.01),
            )
        console.print(f"[bold green]Epoch {epoch + 1} ✔[/bold green]")
        accuracy, test_loss = test(test_dataloader, network, torch.nn.CrossEntropyLoss())
        console.print(
            f"[bold yellow]Test accuracy: {accuracy:.4f}, Test loss: {test_loss:.4f}[/bold yellow]"
        )

    console.print("[bold blue]Training complete![/bold blue]")


if __name__ == "__main__":
    main()
