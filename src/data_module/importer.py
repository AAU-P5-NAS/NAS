import enum
from typing import Optional
import torch
import pandas as pd
from rich.console import Console
from pathlib import Path
import requests
import gdown
# import matplotlib.pyplot as plt

KAGGLE_DEFAULT_PATH: str = "src/data_module/az_images_data.csv"
EMNIST_TEST_PATH: str = "src/data_module/emnist_letters_test.csv"
EMNIST_TRAIN_PATH: str = "src/data_module/emnist_letters_train.csv"
EMNIST_TEST_URL: str = (
    "https://drive.google.com/uc?export=download&id=17vPi4bWNX0q7y7190FF6L9M-szizzYp2"
)
EMNIST_TRAIN_URL: str = (
    "https://drive.google.com/uc?export=download&id=1K4nwU7sqPTkltz7-gtC0ds6IyPdDwh7B"
)
GRAYSCALE_NUM_CHANNELS: int = 1
IMG_DEFAULT_SIZE: tuple[int, int] = (28, 28)
NUM_CLASSES: int = 26
DEFAULT_H: int
DEFAULT_W: int
DEFAULT_H, DEFAULT_W = IMG_DEFAULT_SIZE
console = Console()


class DatasetOption(enum.Enum):
    KAGGLE = 0
    EMNIST_LETTERS = 1


def fetch_csv_from_url(src: str, dest: str):
    """
    Fetches a CSV file from a public Google Drive link and saves it to dest.
    Works for links of the form:
      https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
    """

    response = requests.get(src)
    response.raise_for_status()
    # Ensure destination directory exists
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    gdown.download(src, dest, quiet=True)


def import_kaggle_csv(max_per_class: int | None = None):
    with console.status(f"[bold blue]Loading data from {KAGGLE_DEFAULT_PATH}..."):
        try:
            data_file = pd.read_csv(KAGGLE_DEFAULT_PATH, header=None)
        except FileNotFoundError:
            raise ValueError(
                f"The provided filepath for data importer could not be found: {KAGGLE_DEFAULT_PATH}"
            ) from None

        data = data_file.values.astype("float32")
        labels = torch.tensor(data[:, 0], dtype=torch.long)
        values = torch.tensor(data[:, 1:] / 255.0)
        values = values.view(-1, GRAYSCALE_NUM_CHANNELS, DEFAULT_H, DEFAULT_W)

        if max_per_class is not None:  # Limit the number of samples per class
            selected_indices = []
            for label in labels.unique():
                label_indices = torch.where(labels == label)[0]
                n_select = min(max_per_class, len(label_indices))
                selected_indices.append(label_indices[:n_select])
            selected_indices = torch.cat(selected_indices)

            values = values[selected_indices]
            labels = labels[selected_indices]

        return torch.utils.data.TensorDataset(values, labels)


def import_emnist_letters(max_per_class: int | None = None):
    if not Path(EMNIST_TRAIN_PATH).is_file():
        console.print("[bold orange]EMNIST Letters train data not found.[/bold orange]")
        with console.status("[bold yellow]Downloading EMNIST Letters train data...[/bold yellow]"):
            fetch_csv_from_url(EMNIST_TRAIN_URL, EMNIST_TRAIN_PATH)
            console.print("[bold green]EMNIST Letters train data downloaded ✔[/bold green]")
    if not Path(EMNIST_TEST_PATH).is_file():
        console.print("[bold orange]EMNIST Letters test data not found.[/bold orange]")
        with console.status("[bold yellow]Downloading EMNIST Letters test data...[/bold yellow]"):
            fetch_csv_from_url(EMNIST_TEST_URL, EMNIST_TEST_PATH)
            console.print("[bold green]EMNIST Letters test data downloaded ✔[/bold green]")

    with console.status("[bold blue]Loading EMNIST Letters data...[/bold blue]"):
        try:
            train_file = pd.read_csv(EMNIST_TRAIN_PATH, header=None)
            test_file = pd.read_csv(EMNIST_TEST_PATH, header=None)
        except FileNotFoundError:
            raise ValueError(
                f"The provided filepath for data importer could not be found: {EMNIST_TRAIN_PATH} or {EMNIST_TEST_PATH}"
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
        )

    pass


class DataImporter:
    """
    Imports data from a CSV file and provides DataLoaders for training and testing CNNs.
    Can optionally limit the number of samples per class.
    """

    def __init__(
        self,
        dataset_option: DatasetOption,
        max_per_class: int | None = None,
    ):
        self.dataset_option = dataset_option
        if dataset_option is None:
            raise ValueError("dataset_option must be provided")

        if dataset_option == DatasetOption.KAGGLE:
            console.print("[yellow]Using kaggle dataset.[/yellow]")
            self.dataset = import_kaggle_csv(max_per_class)

        if dataset_option == DatasetOption.EMNIST_LETTERS:
            console.print("[yellow]Using EMNIST Letters dataset.[/yellow]")
            self.train_dataset, self.test_dataset = import_emnist_letters(max_per_class)

        console.print(
            f"[bold green]Data loaded ✔ (classes limited to {max_per_class} samples each)[/bold green]"
            if max_per_class
            else "[bold green]Data loaded ✔[/bold green]"
        )

    def get_as_cnn(
        self, batch_size: int, test_split: float, seed: Optional[int] = None, shuffle: bool = True
    ):
        """
        Returns a DataLoader for both the training_data and test_data, shaped for CNN input.

        :Arguments:
        - batch_size: Batch size for the DataLoaders.
        - test_split: Fraction of data to use as test set (between 0 and 1).
        - seed (optional): Random seed for reproducibility.
        - shuffle (optional): Whether to shuffle the training data. Default is True.

        :Returns:
        - tuple of (train_dataloader, test_dataloader): DataLoaders for training and test data.

        :Raises:
        - ValueError: If batch_size is not a positive integer or if test_split is not between 0 and 1.

        """

        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        if self.dataset_option == DatasetOption.EMNIST_LETTERS:
            train_dataloader = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=batch_size, shuffle=shuffle
            )
            test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size)

            # plot 5 examples from the first batch of the train dataloader
            """ images, labels = next(iter(train_dataloader))
            n = min(20, images.size(0))
            fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
            for i in range(n):
                img = images[i].squeeze().cpu().numpy()
                lbl = labels[i].item()
                ax = axes[i] if n > 1 else axes
                ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
                ax.set_title(chr(ord("A") + int(lbl)))
                ax.axis("off")
            plt.tight_layout()
            plt.show() """
            return train_dataloader, test_dataloader

        if not 0 < test_split < 1:
            raise ValueError("test_split must be between 0 and 1 (exclusive)")

        generator = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()

        train_size = int(len(self.dataset) * (1 - test_split))
        test_size = len(self.dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, test_size], generator=generator
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle
        )
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
        # plot 5 examples from the first batch of the train dataloader
        """ images, labels = next(iter(train_dataloader))
        n = min(20, images.size(0))
        fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
        for i in range(n):
            img = images[i].squeeze().cpu().numpy()
            lbl = labels[i].item()
            ax = axes[i] if n > 1 else axes
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(chr(ord("A") + int(lbl)))
            ax.axis("off")
        plt.tight_layout()
        plt.show() """
        return train_dataloader, test_dataloader
