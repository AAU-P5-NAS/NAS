from typing import Tuple
from rich.console import Console
from data_module.dataset import DatasetOption
from data_module.import_utils import visualize_samples
import torch

console = Console()


class DataImporter:
    """
    Imports data from a CSV file and provides DataLoaders for training and testing CNNs.
    Can optionally limit the number of samples per class.
    Can also visualize sample images from the dataset for inspection.
    """

    def __init__(
        self,
        dataset_option: DatasetOption,
        max_per_class: int | None = None,
    ):
        if dataset_option is None:
            raise ValueError("dataset_option must be provided")
        self.dataset_option = dataset_option
        self.train_dataset, self.test_dataset, self.train_num_classes, self.test_num_classes = (
            dataset_option.import_data(max_per_class)
        )
        self.label_fn = dataset_option.get_label_fn()

        console.print(
            f"[bold green]Data loaded ✔ (classes limited to {max_per_class} samples each)[/bold green]"
            if max_per_class
            else "[bold green]Data loaded ✔[/bold green]"
        )

    def get_dataloaders(self, batch_size: int, shuffle: bool = True):
        """
        Returns a DataLoader for both the training_data and test_data, shaped for CNN input.

        :Arguments:
        - batch_size: Batch size for the DataLoaders.
        - shuffle (optional): Whether to shuffle the training data. Default is True.

        :Returns:
        - tuple of (train_dataloader, test_dataloader): DataLoaders for training and test data.

        :Raises:
        - ValueError: If batch_size is not a positive integer.
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=shuffle
        )
        test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size)

        visualize_samples(train_dataloader, self.dataset_option.get_label_fn(), num_samples=30)

        return train_dataloader, test_dataloader

    def get_num_classes(self) -> Tuple[int, int]:
        """Returns the number of unique classes in the dataset.

        :Returns:
        - tuple of (train_num_classes, test_num_classes): Number of unique classes in training and test datasets.
        """

        return self.train_num_classes, self.test_num_classes
