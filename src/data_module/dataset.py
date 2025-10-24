from torch.utils.data import TensorDataset
from data_module.emnist.emnist_balanced import import_emnist_balanced
from data_module.emnist.emnist_letters import import_emnist_letters
from data_module.emnist.emnist_mnist import import_emnist_mnist
from typing import Callable, Tuple
import enum
import pandas as pd


class DatasetOption(enum.Enum):
    EMNIST_MNIST = 0
    EMNIST_LETTERS = 1
    EMNIST_BALANCED = 2
    EMNIST_BYCLASS = 3
    EMNIST_BYMERGE = 4

    def get_mapping_string(self) -> str:
        """
        Returns the file path to the mapping file for the dataset option.
        Used to map label indices to characters. Only used for visualization purposes.
        """
        match self:
            case DatasetOption.EMNIST_MNIST:
                return "src/data_module/emnist/emnist_mnist_mapping.txt"
            case DatasetOption.EMNIST_LETTERS:
                return "src/data_module/emnist/emnist_letters_mapping.txt"
            case DatasetOption.EMNIST_BALANCED:
                return "src/data_module/emnist/emnist_balanced_mapping.txt"
            case DatasetOption.EMNIST_BYCLASS:
                return "src/data_module/emnist/emnist_byclass_mapping.txt"
            case DatasetOption.EMNIST_BYMERGE:
                return "src/data_module/emnist/emnist_bymerge_mapping.txt"
            case _:
                raise ValueError(f"Unknown dataset option: {self}")

    def get_label_fn(self) -> Callable:
        """
        Returns a function that maps tensor indices to their corresponding label strings.
        Only used for visualization purposes.

        """
        match self:
            case DatasetOption.EMNIST_MNIST:
                return lambda x: str(x)
            case DatasetOption.EMNIST_LETTERS:
                mapping = pd.read_csv(
                    self.get_mapping_string(),
                    sep=r"\s+",  # handles multiple spaces
                    header=None,
                    names=["index", "uppercase_ascii", "lowercase_ascii"],
                )
                index_to_char = {
                    row["index"] - 1: chr(row["uppercase_ascii"]) for _, row in mapping.iterrows()
                }
                return lambda x: index_to_char[x]
            case DatasetOption.EMNIST_BALANCED:
                mapping = pd.read_csv(
                    self.get_mapping_string(), sep=r"\s+", header=None, names=["index", "ascii"]
                )
                index_to_char = {
                    int(row["index"]): chr(int(row["ascii"])) for _, row in mapping.iterrows()
                }
                return lambda x: index_to_char[x]
            case _:
                raise ValueError(f"Unknown dataset option: {self}")

    def import_data(
        self, max_per_class: int | None = None
    ) -> Tuple[TensorDataset, TensorDataset, int, int]:
        """Imports the dataset corresponding to the DatasetOption.

        :Arguments:
        - max_per_class (optional): Maximum number of samples to import per class. If None, imports all samples.

        :Returns:
        - tuple of (train_dataset, test_dataset, train_num_classes, test_num_classes): The imported datasets and their class counts.

        """
        match self:
            case DatasetOption.EMNIST_MNIST:
                return import_emnist_mnist(max_per_class)
            case DatasetOption.EMNIST_LETTERS:
                return import_emnist_letters(max_per_class)
            case DatasetOption.EMNIST_BALANCED:
                return import_emnist_balanced(max_per_class)
            case _:
                raise ValueError(f"Unknown dataset option: {self}")
