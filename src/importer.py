import torch
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt

CSV_DEFAULT_PATH = "./az_images_data.csv"


class ColorMode(Enum):
    GRAYSCALE = "grayscale"
    RGB = "rgb"


class DataImporter:
    def __init__(
        self,
        filepath: str = CSV_DEFAULT_PATH,
        img_size: tuple[int, int] = (28, 28),
        color_mode: ColorMode = ColorMode.GRAYSCALE,
    ):
        self.img_size = img_size
        self.color_mode = color_mode

        data_frame = pd.read_csv(filepath, header=None)
        data = data_frame.values.astype("float32")[:, 1:] / 255.0
        print("Raw data shape:", data.shape)
        print("raw data", data)
        h, w = img_size
        if color_mode == ColorMode.GRAYSCALE:
            self.channels = 1
            expected_cols = h * w
        elif color_mode == ColorMode.RGB:
            self.channels = 3
            expected_cols = 3 * h * w
        else:
            raise ValueError("color_mode must be 'grayscale' or 'rgb'")

        if data.shape[1] != expected_cols:
            raise ValueError(f"CSV should have {expected_cols} columns for {color_mode}")

        self.data = torch.tensor(data)  # always store internally as [N, features]

    def get_as_ffnn(self):
        return self.data

    def get_as_cnn(self, batch_size: int | None = None):
        h, w = self.img_size
        reshaped = self.data.view(-1, self.channels, h, w)
        if batch_size is None:
            return reshaped
        return reshaped.split(batch_size)

    def show_random_sample(self, index: int = 66):
        h, w = self.img_size
        img = self.data[index].view(h, w)
        plt.imshow(img, cmap="gray")
        plt.show()
