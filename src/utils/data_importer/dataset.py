from torch.utils.data import TensorDataset
from src.utils.data_importer.cifar.cifar10 import import_cifar10
from src.utils.data_importer.emnist.emnist_balanced import import_emnist_balanced
from src.utils.data_importer.emnist.emnist_byclass import import_emnist_byclass
from src.utils.data_importer.emnist.emnist_bymerge import import_emnist_bymerge
from src.utils.data_importer.emnist.emnist_letters import import_emnist_letters
from src.utils.data_importer.emnist.emnist_mnist import import_emnist_mnist
from typing import Callable, Tuple
import enum
import pandas as pd


class DatasetOption(enum.Enum):
    EMNIST_MNIST = 0
    EMNIST_LETTERS = 1
    EMNIST_BALANCED = 2
    EMNIST_BYCLASS = 3
    EMNIST_BYMERGE = 4
    CIFAR_10 = 5

    def get_label_fn(self) -> Callable:
        """
        Returns a function that maps label indices to their corresponding label strings in ascii.
        Only used for visualization purposes.

        """
        match self:
            case DatasetOption.EMNIST_MNIST:
                return lambda x: str(x)
            case DatasetOption.EMNIST_LETTERS:
                return three_column_mapping_func(self)
            case DatasetOption.CIFAR_10:
                return cifar_label_mapping_func()
            case _:
                return two_column_mapping_func(self)

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
            case DatasetOption.EMNIST_BYCLASS:
                return import_emnist_byclass(max_per_class)
            case DatasetOption.EMNIST_BYMERGE:
                return import_emnist_bymerge(max_per_class)
            case DatasetOption.CIFAR_10:
                return import_cifar10(max_per_class)
            case _:
                raise ValueError(f"Unknown dataset option: {self}")


def three_column_mapping_func(dataset: DatasetOption):
    mapping = pd.read_csv(
        get_mapping_string(dataset),
        sep=r"\s+",
        header=None,
        names=["index", "uppercase_ascii", "lowercase_ascii"],
    )
    index_to_char = {row["index"] - 1: chr(row["uppercase_ascii"]) for _, row in mapping.iterrows()}
    return lambda x: index_to_char[x]


def two_column_mapping_func(dataset: DatasetOption):
    mapping = pd.read_csv(
        get_mapping_string(dataset), sep=r"\s+", header=None, names=["index", "ascii"]
    )
    index_to_char = {int(row["index"]): chr(int(row["ascii"])) for _, row in mapping.iterrows()}
    return lambda x: index_to_char[x]


def get_mapping_string(dataset_option: DatasetOption) -> str:
    """
    Returns the file path to the mapping file for the dataset option.
    Used to map label indices to characters. Only used for visualization purposes.
    """
    match dataset_option:
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
            raise ValueError(f"Unknown dataset option: {dataset_option}")


def cifar_label_mapping_func():
    index_to_label = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }
    return lambda x: index_to_label[x]
