from typing import Union, overload
import torch
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from pydantic import BaseModel, ConfigDict

CSV_DEFAULT_PATH = "src/data_module/az_images_data.csv"
GRAYSCALE_NUM_CHANNELS = 1


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


class DataImporter:
    """
     Imports data from a CSV file and provides methods to reshape it for FFNN or CNN input

     :Arguments:
     - filepath: Path to the CSV file containing the data. Default is "src/az_images_data.csv".
     - img_size: Tuple specifying the height and width of the images. Default is (28, 28).

    :Methods:
     - get_as_ffnn(): Returns two DataLoaders (train_loader, test_loader) for FFNN input with a user-configurable split.
     - get_as_cnn(batch_size=None, conv_args=None): Returns the data reshaped for CNN input, optionally split into batches. If conv_args is provided, also returns a Conv2d layer initialized with those arguments.
     - show_random_ffn_sample(index=None): Displays a random sample from the FFNN data as a 2D image.
     - show_random_conv_sample(conv_data, index=0, channel=0): Displays a sample from the convoluted data as a 2D image.
    """

    def __init__(
        self,
        filepath: str = CSV_DEFAULT_PATH,
        img_size: tuple[int, int] = (28, 28),
    ):
        console = Console()
        with console.status(f"[bold blue]Loading data from {filepath}..."):
            self.img_size = img_size
            data_frame = pd.read_csv(filepath, header=None)
            data = data_frame.values.astype("float32")
            labels = torch.tensor(data[:, 0], dtype=torch.long)  # [N]
            features = torch.tensor(data[:, 1:] / 255.0)  # [N, features]
            features = features.view(
                -1, GRAYSCALE_NUM_CHANNELS, img_size[0], img_size[1]
            )  # [N, 1, H, W]
            self.dataset = torch.utils.data.TensorDataset(features, labels)
            self.data = features
            self.labels = labels
        console.print("[bold green]Data loaded ✔[/bold green]")

    def get_as_ffnn(
        self,
        batch_size: int | None = None,
        shuffle: bool = True,
        test_split: float = 0.2,
        random_seed: int | None = None,
    ):
        """
        Returns two DataLoaders (train_loader, test_loader) for FFNN input.
        The split is disjoint and user-configurable.

        :Arguments:
        - batch_size: Batch size for the DataLoaders.
        - shuffle: Whether to shuffle the training data.
        - test_split: Fraction of data to use as test set (between 0 and 1).
        - random_seed: Optional random seed for reproducibility.

        :Returns:
        - train_loader: DataLoader for training data.
        - test_loader: DataLoader for test data.
        """
        if not (0 < test_split < 1):
            raise ValueError("test_split must be between 0 and 1 (exclusive).")
        dataset_size = len(self.dataset)
        test_size = int(dataset_size * test_split)
        train_size = dataset_size - test_size

        if batch_size is None or batch_size <= 0:
            batch_size = 1

        if random_seed is not None:
            generator = torch.Generator().manual_seed(random_seed)
        else:
            generator = None

        train_dataset, test_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, test_size], generator=generator
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        return train_loader, test_loader

    # Overloads for get_as_cnn to clarify return types
    @overload
    def get_as_cnn(
        self, batch_size: None = None, conv_args: None = None
    ) -> torch.utils.data.DataLoader: ...
    @overload
    def get_as_cnn(
        self, batch_size: int, conv_args: None = None
    ) -> torch.utils.data.DataLoader: ...
    @overload
    def get_as_cnn(
        self, batch_size: None = None, conv_args: ConvolutionalArguments = ...
    ) -> tuple[torch.utils.data.DataLoader, torch.nn.Conv2d]: ...
    @overload
    def get_as_cnn(
        self, batch_size: int, conv_args: ConvolutionalArguments = ...
    ) -> tuple[torch.utils.data.DataLoader, torch.nn.Conv2d]: ...
    def get_as_cnn(
        self, batch_size: int | None = None, conv_args: ConvolutionalArguments | None = None
    ) -> Union[
        torch.utils.data.DataLoader,
        tuple[torch.utils.data.DataLoader, torch.nn.Conv2d],
    ]:
        """
        Returns a DataLoader for the data reshaped for CNN input.
        If conv_args is provided, also returns a Conv2d layer initialized with those arguments.

        - If conv_args is None: returns DataLoader[(N, C, H, W), label]
        - If conv_args is given: returns (DataLoader, Conv2d layer)

        The Conv2d layer is not used in the DataLoader, but is provided for convenience
        so the user can immediately apply it to batches from the DataLoader.
        """
        h, w = self.img_size
        cnn_data = self.data.view(-1, GRAYSCALE_NUM_CHANNELS, h, w)
        dataset = torch.utils.data.TensorDataset(cnn_data, self.labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        if conv_args is None:
            return dataloader

        conv = torch.nn.Conv2d(
            in_channels=conv_args.in_channels,
            out_channels=conv_args.out_channels,
            kernel_size=conv_args.kernel_size,
            stride=conv_args.stride,
            padding=conv_args.padding,
            dilation=conv_args.dilation,
            groups=conv_args.groups,
            bias=conv_args.bias,
            padding_mode=conv_args.padding_mode,
            device=conv_args.device,
            dtype=conv_args.dtype,
        )
        return dataloader, conv

    def show_random_ffn_sample(self, index: int | None = None):
        """
        Displays a random sample from the FFNN data as a 2D image.

        :Arguments:
        - index (optional): Index of the sample to display. If None, a random sample is chosen.

        :Returns: None. Displays the image using matplotlib.
        """
        h, w = self.img_size
        rand_num = int(torch.randint(0, len(self.data), (1,)).item())
        if index is None:
            index = rand_num
        if index is None or index >= len(self.data):
            index = len(self.data) - 1
        img = self.data[index if index > 0 else 0].view(h, w)
        plt.imshow(img, cmap="gray")
        plt.show()

    def show_random_conv_sample(
        self, conv_data: torch.Tensor, index: int | None = None, channel: int | None = None
    ):
        """
        Displays a sample from the convoluted data as a 2D image.

        :Arguments:
        - conv_data:  The output tensor from a convolutional layer.
        - index (optional): Index of the sample in the batch to display.
        - channel (optional): Channel of the convoluted data to display.

        :Returns: None. Displays the image using matplotlib.
        """
        # If input is [C_out, H, W], add a fake batch dimension
        if conv_data.ndim == 3:
            conv_data = conv_data.unsqueeze(0)  # [1, C_out, H, W]
        if index is None or index >= conv_data.shape[0]:
            index = int(torch.randint(0, conv_data.shape[0], (1,)).item())
        if channel is None or channel >= conv_data.shape[1]:
            channel = int(torch.randint(0, conv_data.shape[1], (1,)).item())
        img = conv_data[index, channel].detach().cpu()
        plt.imshow(img, cmap="gray")
        plt.title(f"Sample {index}, Channel {channel}")
        plt.axis("off")
        plt.show()
