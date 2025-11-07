from pathlib import Path
from typing import Callable
from matplotlib import pyplot as plt
import numpy as np
import requests
import gdown
import torch

# emnist has already split up the data into train and test sets, for valid comparison we use those
GRAYSCALE_NUM_CHANNELS: int = 1
IMG_DEFAULT_SIZE: tuple[int, int] = (28, 28)
DEFAULT_H: int
DEFAULT_W: int
DEFAULT_H, DEFAULT_W = IMG_DEFAULT_SIZE


def transform_google_drive_url_to_direct_download(url: str) -> str:
    """Transforms a Google Drive shareable link into a direct download link."""
    if "drive.google.com" not in url:
        raise ValueError("The provided URL is not a valid Google Drive link.")

    if "id=" in url:
        file_id = url.split("id=")[1].split("&")[0]
    else:
        parts = url.split("/")
        try:
            file_id_index = parts.index("d") + 1
            file_id = parts[file_id_index]
        except (ValueError, IndexError):
            raise ValueError("Could not extract file ID from the provided Google Drive URL.")

    direct_download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    return direct_download_url


def fetch_dataset_from_url(src: str, dest: str):
    """Fetches a dataset from a public Google Drive link and saves it to dest"""
    download_src = transform_google_drive_url_to_direct_download(src)
    response = requests.get(download_src)
    response.raise_for_status()
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure destination directory exists

    gdown.download(download_src, dest, quiet=True)


def visualize_samples(
    dataloader: torch.utils.data.DataLoader, label_fn: Callable, num_samples: int = 9
):
    images, labels = next(iter(dataloader))
    n = min(num_samples, images.size(0))

    # compute grid size (instead of showing long line of images)
    rows = int(n**0.5) or 1
    cols = int(np.ceil(n / rows))

    _, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.array(axes).reshape(-1)

    for i in range(n):
        img = images[i].squeeze().cpu().numpy()
        lbl = labels[i].item()
        ax = axes[i]
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(label_fn(lbl))
        ax.axis("off")

    # turn of unused subplots, if num_samples is not perfect square
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()



def visualize_cifar_samples(
    dataloader: torch.utils.data.DataLoader, label_fn: Callable, num_samples: int = 9
):
    images, labels = next(iter(dataloader))
    n = min(num_samples, images.size(0))

    rows = int(n**0.5) or 1
    cols = int(np.ceil(n / rows))

    _, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.array(axes).reshape(-1)

    for i in range(n):
        img = images[i].permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
        lbl = labels[i].item()
        ax = axes[i]
        ax.imshow(img)
        ax.set_title(label_fn(lbl))
        ax.axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)
    plt.close()