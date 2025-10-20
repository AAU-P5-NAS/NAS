import enum
from typing import Optional
import torch
import pandas as pd
from rich.console import Console

KAGGLE_DEFAULT_PATH: str = "src/data_module/az_images_data.csv"
EMNIST_DEFAULT_PATH: str = "src/data_module/emnist_letters.csv"
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

        return values, labels, torch.utils.data.TensorDataset(values, labels)


def import_emnist_letters(max_per_class: int | None = None):
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
        if dataset_option is None:
            raise ValueError("dataset_option must be provided")

        if dataset_option == DatasetOption.KAGGLE:
            console.print("[yellow]Using kaggle dataset.[/yellow]")
            self.data, self.labels, self.dataset = import_kaggle_csv(max_per_class)

        if dataset_option == DatasetOption.EMNIST_LETTERS:
            console.print("[yellow]Using EMNIST Letters dataset.[/yellow]")
            # self.data, self.labels, self.dataset = import_emnist_letters(max_per_class)

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
        return train_dataloader, test_dataloader
