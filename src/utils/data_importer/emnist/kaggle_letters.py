import torch
from rich.console import Console
import pandas as pd

from src.utils.data_importer.import_utils import DEFAULT_H, DEFAULT_W, GRAYSCALE_NUM_CHANNELS

KAGGLE_DEFAULT_PATH: str = "src/utils/data_importer/az_images_data.csv"
console = Console()


def import_kaggle_csv(max_per_class: int | None = None):
    with console.status(f"[bold blue]Loading data from {KAGGLE_DEFAULT_PATH}..."):
        try:
            data_file = pd.read_csv(KAGGLE_DEFAULT_PATH, header=None)
        except FileNotFoundError:
            raise ValueError(f"Provided filepath not found: {KAGGLE_DEFAULT_PATH}") from None

        data = data_file.values.astype("float64")
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
