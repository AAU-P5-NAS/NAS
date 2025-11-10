from pathlib import Path
import pandas as pd
import torch
from rich.console import Console
from src.data_module.import_utils import (
    fetch_dataset_from_url,
    GRAYSCALE_NUM_CHANNELS,
    DEFAULT_H,
    DEFAULT_W,
)

console = Console()


EMNIST_MNIST_TEST_PATH: str = "src/data_module/emnist/emnist_mnist_test.csv"
EMNIST_MNIST_TRAIN_PATH: str = "src/data_module/emnist/emnist_mnist_train.csv"
EMNIST_MNIST_MAPPING_PATH: str = "src/data_module/emnist/emnist_mnist_mapping.txt"
EMNIST_MNIST_TEST_URL: str = (
    "https://drive.google.com/file/d/1E6UT193I2KWPa6wnobDg2FpAR51zL657/view?usp=drive_link"
)
EMNIST_MNIST_TRAIN_URL: str = (
    "https://drive.google.com/file/d/1EVcbEjfLGXihqvbRkQnza8CTey6YgSQu/view?usp=drive_link"
)
EMNIST_MNIST_MAPPING_URL: str = (
    "https://drive.google.com/file/d/1j7dw1RV6mwXFLrKJ2t4zWB9hmU9BVuns/view?usp=drive_link"
)


def import_emnist_mnist(max_per_class: int | None = None):
    if not Path(EMNIST_MNIST_TRAIN_PATH).is_file():
        console.print("[bold yellow]EMNIST MNIST train data not found.[/bold yellow]")
        with console.status("[bold yellow]Downloading EMNIST MNIST train data...[/bold yellow]"):
            fetch_dataset_from_url(EMNIST_MNIST_TRAIN_URL, EMNIST_MNIST_TRAIN_PATH)
            console.print("[bold green]EMNIST MNIST train data downloaded ✔[/bold green]")
    else:
        console.print("[bold green]EMNIST MNIST train data found ✔[/bold green]")

    if not Path(EMNIST_MNIST_TEST_PATH).is_file():
        console.print("[bold yellow]EMNIST MNIST test data not found.[/bold yellow]")
        with console.status("[bold yellow]Downloading EMNIST MNIST test data...[/bold yellow]"):
            fetch_dataset_from_url(EMNIST_MNIST_TEST_URL, EMNIST_MNIST_TEST_PATH)
            console.print("[bold green]EMNIST MNIST test data downloaded ✔[/bold green]")
    else:
        console.print("[bold green]EMNIST MNIST test data found ✔[/bold green]")

    console.print("[bold green]EMNIST MNIST Mapping not necessary. ✔[/bold green]")

    with console.status("[bold blue]Loading EMNIST MNIST data...[/bold blue]"):
        try:
            train_file = pd.read_csv(EMNIST_MNIST_TRAIN_PATH, header=None)
            test_file = pd.read_csv(EMNIST_MNIST_TEST_PATH, header=None)
        except FileNotFoundError:
            raise ValueError(
                f"The provided filepath for data importer could not be found: {EMNIST_MNIST_TRAIN_PATH} or {EMNIST_MNIST_TEST_PATH}"
            ) from None

        train_data = train_file.values.astype("float32")
        test_data = test_file.values.astype("float32")

        train_labels = torch.tensor(train_data[:, 0], dtype=torch.long)
        test_labels = torch.tensor(test_data[:, 0], dtype=torch.long)

        train_values = torch.tensor(train_data[:, 1:] / 255.0)
        test_values = torch.tensor(test_data[:, 1:] / 255.0)

        train_values = train_values.view(
            -1, GRAYSCALE_NUM_CHANNELS, DEFAULT_H, DEFAULT_W
        ).transpose(2, 3)
        test_values = test_values.view(-1, GRAYSCALE_NUM_CHANNELS, DEFAULT_H, DEFAULT_W).transpose(
            2, 3
        )

        if max_per_class is not None:  # Limit the number of samples per class
            selected_indices = []
            for label in train_labels.unique():
                label_indices = torch.where(train_labels == label)[0]
                n_select = min(max_per_class, len(label_indices))
                selected_indices.append(label_indices[:n_select])
            selected_indices = torch.cat(selected_indices)

            train_values = train_values[selected_indices]
            train_labels = train_labels[selected_indices]

        return (
            torch.utils.data.TensorDataset(train_values, train_labels),
            torch.utils.data.TensorDataset(test_values, test_labels),
            len(torch.unique(train_labels)),
            len(torch.unique(test_labels)),
        )
