from pathlib import Path
import pandas as pd
import torch
from rich.console import Console
from data_module.import_utils import (
    DEFAULT_H,
    DEFAULT_W,
    GRAYSCALE_NUM_CHANNELS,
    fetch_dataset_from_url,
)

console = Console()

EMNIST_LETTERS_TEST_PATH: str = "src/data_module/emnist/emnist_letters_test.csv"
EMNIST_LETTERS_TRAIN_PATH: str = "src/data_module/emnist/emnist_letters_train.csv"
EMNIST_TEST_URL: str = (
    "https://drive.google.com/file/d/17vPi4bWNX0q7y7190FF6L9M-szizzYp2/view?usp=drive_link"
)
EMNIST_TRAIN_URL: str = (
    "https://drive.google.com/file/d/1K4nwU7sqPTkltz7-gtC0ds6IyPdDwh7B/view?usp=drive_link"
)
EMNIST_MAPPING_PATH: str = "src/data_module/emnist/emnist_letters_mapping.txt"
EMNIST_MAPPING_URL: str = (
    "https://drive.google.com/file/d/1FOEMfym9mLCFijxpA9RWCbFysnU3UfUE/view?usp=drive_link"
)


def import_emnist_letters(max_per_class: int | None = None):
    if not Path(EMNIST_LETTERS_TRAIN_PATH).is_file():
        console.print("[bold yellow]EMNIST Letters train data not found.[/bold yellow]")
        with console.status("[bold yellow]Downloading EMNIST Letters train data...[/bold yellow]"):
            fetch_dataset_from_url(EMNIST_TRAIN_URL, EMNIST_LETTERS_TRAIN_PATH)
            console.print("[bold green]EMNIST Letters train data downloaded ✔[/bold green]")
    if not Path(EMNIST_LETTERS_TEST_PATH).is_file():
        console.print("[bold yellow]EMNIST Letters test data not found.[/bold yellow]")
        with console.status("[bold yellow]Downloading EMNIST Letters test data...[/bold yellow]"):
            fetch_dataset_from_url(EMNIST_TEST_URL, EMNIST_LETTERS_TEST_PATH)
            console.print("[bold green]EMNIST Letters test data downloaded ✔[/bold green]")

    if not Path(EMNIST_MAPPING_PATH).is_file():
        console.print("[bold yellow]EMNIST Letters Mapping not found.[/bold yellow]")
        with console.status("[bold yellow]Downloading EMNIST Letters Mapping...[/bold yellow]"):
            fetch_dataset_from_url(EMNIST_MAPPING_URL, EMNIST_MAPPING_PATH)
            console.print("[bold green]EMNIST Letters Mapping downloaded ✔[/bold green]")

    with console.status("[bold blue]Loading EMNIST Letters data...[/bold blue]"):
        try:
            train_file = pd.read_csv(EMNIST_LETTERS_TRAIN_PATH, header=None)
            test_file = pd.read_csv(EMNIST_LETTERS_TEST_PATH, header=None)
        except FileNotFoundError:
            raise ValueError(
                f"The provided filepath for data importer could not be found: {EMNIST_LETTERS_TRAIN_PATH} or {EMNIST_LETTERS_TEST_PATH}"
            ) from None

        train_data = train_file.values.astype("float32")
        test_data = test_file.values.astype("float32")

        train_labels = torch.tensor(train_data[:, 0] - 1, dtype=torch.long)
        test_labels = torch.tensor(test_data[:, 0] - 1, dtype=torch.long)

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
