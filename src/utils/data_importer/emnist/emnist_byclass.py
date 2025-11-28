from pathlib import Path
import pandas as pd
import torch
from rich.console import Console

from src.utils.data_importer.import_utils import (
    fetch_dataset_from_url,
    GRAYSCALE_NUM_CHANNELS,
    DEFAULT_H,
    DEFAULT_W,
)

console = Console()

EMNIST_BYCLASS_TEST_PATH: str = "src/utils/data_importer/emnist/emnist_byclass_test.csv"
EMNIST_BYCLASS_TRAIN_PATH: str = "src/utils/data_importer/emnist/emnist_byclass_train.csv"
EMNIST_BYCLASS_TEST_URL: str = (
    "https://drive.google.com/file/d/1a6te_dwNlUDPjmqsn6DyqjZ5KMHYVVWL/view?usp=drive_link"
)
EMNIST_BYCLASS_TRAIN_URL: str = (
    "https://drive.google.com/file/d/1K2HDLMDO6WiG0TOi_ntlGbKCeQud_BWE/view?usp=drive_link"
)
EMNIST_BYCLASS_MAPPING_PATH: str = "src/utils/data_importer/emnist/emnist_byclass_mapping.txt"
EMNIST_BYCLASS_MAPPING_URL: str = (
    "https://drive.google.com/file/d/1rkwVPW1Ui8y_p--Br_3xZtVEjkB3cdg-/view?usp=drive_link"
)


def import_emnist_byclass(max_per_class: int | None = None):
    if not Path(EMNIST_BYCLASS_TRAIN_PATH).is_file():
        console.print("[bold yellow]EMNIST ByClass train data not found.[/bold yellow]")
        response = (
            input(
                "The EMNIST ByClass dataset is very large (1.25gb). Do you want to download the train data? (yes/no): "
            )
            .strip()
            .lower()
        )
        if response == "yes":
            with console.status(
                "[bold yellow]Downloading EMNIST ByClass train data...[/bold yellow]"
            ):
                fetch_dataset_from_url(EMNIST_BYCLASS_TRAIN_URL, EMNIST_BYCLASS_TRAIN_PATH)
                console.print("[bold green]EMNIST ByClass train data downloaded ✔[/bold green]")
        else:
            console.print("[bold red]Aborting. train data is required.[/bold red]")
            exit(1)

    if not Path(EMNIST_BYCLASS_TEST_PATH).is_file():
        console.print("[bold yellow]EMNIST ByClass test data not found.[/bold yellow]")
        with console.status("[bold yellow]Downloading EMNIST ByClass test data...[/bold yellow]"):
            fetch_dataset_from_url(EMNIST_BYCLASS_TEST_URL, EMNIST_BYCLASS_TEST_PATH)
            console.print("[bold green]EMNIST ByClass test data downloaded ✔[/bold green]")

    if not Path(EMNIST_BYCLASS_MAPPING_PATH).is_file():
        console.print("[bold yellow]EMNIST ByClass Mapping not found.[/bold yellow]")
        with console.status("[bold yellow]Downloading EMNIST ByClass Mapping...[/bold yellow]"):
            fetch_dataset_from_url(EMNIST_BYCLASS_MAPPING_URL, EMNIST_BYCLASS_MAPPING_PATH)
            console.print("[bold green]EMNIST ByClass Mapping downloaded ✔[/bold green]")

    with console.status("[bold blue]Loading EMNIST ByClass data...[/bold blue]"):
        try:
            train_file = pd.read_csv(EMNIST_BYCLASS_TRAIN_PATH, header=None)
            test_file = pd.read_csv(EMNIST_BYCLASS_TEST_PATH, header=None)
        except FileNotFoundError:
            raise ValueError(
                f"The provided filepath for data importer could not be found: {EMNIST_BYCLASS_TRAIN_PATH} or {EMNIST_BYCLASS_TEST_PATH}"
            ) from None

        train_data = train_file.values.astype("float64")
        test_data = test_file.values.astype("float64")

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

        print("Train labels:", train_labels.min(), train_labels.max())
        print("Test labels:", test_labels.min(), test_labels.max())
        print("Number of classes:", len(torch.unique(train_labels)), len(torch.unique(test_labels)))

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
