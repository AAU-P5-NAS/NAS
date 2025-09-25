from typing import Optional
import torch
import pandas as pd
from rich.console import Console
from pydantic import BaseModel, ConfigDict

CSV_DEFAULT_PATH: str = "src/data_module/az_images_data.csv"
GRAYSCALE_NUM_CHANNELS: int = 1
IMG_DEFAULT_SIZE: tuple[int, int] = (28, 28)
DEFAULT_H: int
DEFAULT_W: int
DEFAULT_H, DEFAULT_W = IMG_DEFAULT_SIZE
console = Console()


class ConvolutionalArguments(BaseModel):
    in_channels: int = 1
    out_channels: int = 32
    kernel_size: int | tuple[int, int] = (3, 3)
    stride: int = 1
    padding: int = 0
    dilation: int = 1
    groups: int = 1
    bias: bool = True
    padding_mode: str = "zeros"
    device: str | None = None
    dtype: torch.dtype | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class CSVFilepathDoesntExist(Exception):
    def __init__(self, data):
        msg = f"The provided filepath '{data}' does not exist."
        super().__init__(msg)
        self.data = data


class DataImporter:
    """
     Imports data from a CSV file and provides DataLoaders for training and testing CNNs.

    :Methods:
     - get_as_cnn(batch_size=int, test_split=float, seed=Optional[int] = None):
    """

    def __init__(
        self,
        filepath: str = CSV_DEFAULT_PATH,
    ):
        with console.status(f"[bold blue]Loading data from {filepath}..."):
            try:
                data_file = pd.read_csv(filepath, header=None)
            except FileNotFoundError:
                raise CSVFilepathDoesntExist(filepath) from None
            data = data_file.values.astype("float32")

            labels = torch.tensor(data[:, 0], dtype=torch.long)  # [N]
            values = torch.tensor(data[:, 1:] / 255.0)  # [N, features]
            values = values.view(-1, GRAYSCALE_NUM_CHANNELS, DEFAULT_H, DEFAULT_W)  # [N, 1, H, W]

            self.data = values
            self.labels = labels
            self.dataset = torch.utils.data.TensorDataset(values, labels)
        console.print("[bold green]Data loaded ✔[/bold green]")

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
