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


EMNIST_BYMERGE_TEST_PATH: str = "src/utils/data_importer/emnist/emnist_bymerge_test.csv"
EMNIST_BYMERGE_TRAIN_PATH: str = "src/utils/data_importer/emnist/emnist_bymerge_train.csv"
EMNIST_BYMERGE_TEST_URL: str = (
    "https://drive.google.com/file/d/1MBVuTQGXWUhDVinfPDkLiJAxARANos1m/view?usp=drive_link"
)
EMNIST_BYMERGE_TRAIN_URL: str = (
    "https://drive.google.com/file/d/1SBFa4tabL3APvCAUqwpm4F8VOGmW6n8t/view?usp=drive_link"
)
EMNIST_BYMERGE_MAPPING_PATH: str = "src/utils/data_importer/emnist/emnist_bymerge_mapping.txt"
EMNIST_BYMERGE_MAPPING_URL: str = (
    "https://drive.google.com/file/d/1EZzalwy1KaNT5GBtfsxn0R3eCdqz4lfQ/view?usp=drive_link"
)


def import_emnist_bymerge(max_per_class: int | None = None):
    if not Path(EMNIST_BYMERGE_TRAIN_PATH).is_file():
        console.print("[bold yellow]EMNIST ByMerge train data not found.[/bold yellow]")
        response = (
            input(
                "The EMNIST ByMerge dataset is very large (1.26gb). Do you want to download the train data? (yes/no): "
            )
            .strip()
            .lower()
        )
        if response == "yes":
            with console.status(
                "[bold yellow]Downloading EMNIST ByMerge train data...[/bold yellow]"
            ):
                fetch_dataset_from_url(EMNIST_BYMERGE_TRAIN_URL, EMNIST_BYMERGE_TRAIN_PATH)
                console.print("[bold green]EMNIST ByMerge train data downloaded ✔[/bold green]")
        else:
            console.print("[bold red]Aborting. train data is required.[/bold red]")
            exit(1)

    if not Path(EMNIST_BYMERGE_TEST_PATH).is_file():
        console.print("[bold yellow]EMNIST ByMerge test data not found.[/bold yellow]")
        with console.status("[bold yellow]Downloading EMNIST ByMerge test data...[/bold yellow]"):
            fetch_dataset_from_url(EMNIST_BYMERGE_TEST_URL, EMNIST_BYMERGE_TEST_PATH)
            console.print("[bold green]EMNIST ByMerge test data downloaded ✔[/bold green]")

    if not Path(EMNIST_BYMERGE_MAPPING_PATH).is_file():
        console.print("[bold yellow]EMNIST ByMerge Mapping not found.[/bold yellow]")
        with console.status("[bold yellow]Downloading EMNIST ByMerge Mapping...[/bold yellow]"):
            fetch_dataset_from_url(EMNIST_BYMERGE_MAPPING_URL, EMNIST_BYMERGE_MAPPING_PATH)
            console.print("[bold green]EMNIST ByMerge Mapping downloaded ✔[/bold green]")

    with console.status("[bold blue]Loading EMNIST ByMerge data...[/bold blue]"):
        try:
            train_file = pd.read_csv(EMNIST_BYMERGE_TRAIN_PATH, header=None)
            test_file = pd.read_csv(EMNIST_BYMERGE_TEST_PATH, header=None)
        except FileNotFoundError:
            raise ValueError(
                f"The provided filepath for data importer could not be found: {EMNIST_BYMERGE_TRAIN_PATH} or {EMNIST_BYMERGE_TEST_PATH}"
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
