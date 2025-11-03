import pickle
from pathlib import Path
import torch
from torch.utils.data import TensorDataset
import numpy as np
from rich.console import Console

from data_module.import_utils import fetch_dataset_from_url

console = Console()

CIFAR10_DEST = "src/data_module/cifar"
CIFAR10_BATCH_URLS = {
    1: "https://drive.google.com/file/d/1ahjxkJFq7Xd55lRy-1HfZ_hI0vWDCrBC/view?usp=drive_link",  # All batches are in the same archive
    2: "https://drive.google.com/file/d/151BSAdwy-jgcVqzB_FBhqtuoBytnj9kE/view?usp=drive_link",
    3: "https://drive.google.com/file/d/1xa0M1RZHSAffRT1ytYVMpy7oOkxP1cXO/view?usp=drive_link",
    4: "https://drive.google.com/file/d/1Q0Qe5zQPlxojlmF3k_2Eu2R6ewWNm6U0/view?usp=drive_link",
    5: "https://drive.google.com/file/d/1N4l4EeOubRZwuhHOsrLOc_vD9EJdvMtm/view?usp=drive_link",
    "test": "https://drive.google.com/file/d/16iQVSmxZf2jCM-c3SXeAACEjUKcvaHJf/view?usp=drive_link",
}

DEFAULT_H = 32
DEFAULT_W = 32
NUM_CHANNELS = 3  # RGB


def unpickle(file):
    with open(file, "rb") as fo:
        return pickle.load(fo, encoding="bytes")


def import_cifar10(max_per_class: int | None = None):
    train_batches = []
    train_labels = []

    # Iterate over keys in CIFAR10_BATCH_URLS
    for key in CIFAR10_BATCH_URLS:
        if key == "test":
            batch_path = Path(CIFAR10_DEST) / "test_batch"
        else:
            batch_path = Path(CIFAR10_DEST) / f"data_batch_{key}"

        if not batch_path.is_file():
            console.print(f"[bold yellow]{batch_path} not found.[/bold yellow]")
            with console.status(f"[bold yellow]Downloading {batch_path}...[/bold yellow]"):
                fetch_dataset_from_url(CIFAR10_BATCH_URLS[key], str(batch_path))
                console.print(f"[bold green]{batch_path} downloaded ✔[/bold green]")

        batch = unpickle(batch_path)
        if key == "test":
            test_batch = batch
        else:
            train_batches.append(batch[b"data"])
            train_labels.extend(batch[b"labels"])

    # Stack training data
    train_data = np.vstack(train_batches).astype("float32") / 255.0  # normalize
    train_labels = torch.tensor(train_labels, dtype=torch.long)

    # Reshape: N x 3072 -> N x 3 x 32 x 32
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_data = train_data.view(-1, NUM_CHANNELS, DEFAULT_H, DEFAULT_W)

    # Load test batch
    test_data = torch.tensor(test_batch[b"data"], dtype=torch.float32) / 255.0
    test_labels = torch.tensor(test_batch[b"labels"], dtype=torch.long)
    test_data = test_data.view(-1, NUM_CHANNELS, DEFAULT_H, DEFAULT_W)

    # Optional: limit samples per class
    if max_per_class is not None:
        selected_indices = []
        for label in train_labels.unique():
            label_indices = torch.where(train_labels == label)[0]
            n_select = min(max_per_class, len(label_indices))
            selected_indices.append(label_indices[:n_select])
        selected_indices = torch.cat(selected_indices)
        train_data = train_data[selected_indices]
        train_labels = train_labels[selected_indices]

    return (
        TensorDataset(train_data, train_labels),
        TensorDataset(test_data, test_labels),
        len(torch.unique(train_labels)),
        len(torch.unique(test_labels)),
    )
