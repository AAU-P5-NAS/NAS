from typing import Union, overload
import torch
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from pydantic import BaseModel, ConfigDict

CSV_DEFAULT_PATH = "src/az_images_data.csv"
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
     - get_as_ffnn(): Returns the data as a 2D tensor suitable for FFNN input.
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
            data = data_frame.values.astype("float32")[:, 1:] / 255.0
            self.data = torch.tensor(data)  # always store internally as [N, features]

            if len(self.data) <= 783:
                raise ValueError(f"Data has too few features: {len(self.data)} < 784")
        console.print("[bold green]Data loaded ✔[/bold green]")

    def get_as_ffnn(self):
        """
        :Returns: The data as a 2D tensor suitable for FFNN input.
        """
        return self.data

    # Correctly typed overloads for get_as_cnn as return types differ depending on arguments
    @overload
    def get_as_cnn(self, batch_size: None = None, conv_args: None = None) -> torch.Tensor: ...
    @overload
    def get_as_cnn(self, batch_size: int, conv_args: None = None) -> tuple[torch.Tensor, ...]: ...
    @overload
    def get_as_cnn(
        self, batch_size: None = None, conv_args: ConvolutionalArguments = ...
    ) -> tuple[torch.Tensor, torch.nn.Conv2d]: ...
    @overload
    def get_as_cnn(
        self, batch_size: int, conv_args: ConvolutionalArguments = ...
    ) -> tuple[tuple[torch.Tensor, ...], torch.nn.Conv2d]: ...
    def get_as_cnn(
        self, batch_size: int | None = None, conv_args: ConvolutionalArguments | None = None
    ) -> Union[
        torch.Tensor,
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, torch.nn.Conv2d],
        tuple[tuple[torch.Tensor, ...], torch.nn.Conv2d],
    ]:
        """
        Returns the data reshaped for CNN input, optionally split into batches.\n
        If conv_args is provided, also returns a Conv2d layer initialized with those arguments.\n
        \n
        How to use (scroll down):
        1. Without batching and without Conv2d layer:\n
            cnn_data = importer.get_as_cnn()
        2. With batching and without Conv2d layer:\n
            batched_cnn_data = importer.get_as_cnn(batch_size=32)
        3. Without batching and with Conv2d layer:\n
            raw_data, conv_layer = importer.get_as_cnn(conv_args=conv_args)
            convoluted_data = conv_layer(raw_data)
        4. With batching and with Conv2d layer:\n
            batched_raw_data, conv_layer = importer.get_as_cnn(batch_size=32, conv_args=conv_args)
            convoluted_batches = [conv_layer(batch) for batch in batched_raw_data]
        """
        h, w = self.img_size
        reshaped = self.data.view(-1, GRAYSCALE_NUM_CHANNELS, h, w)
        if batch_size is not None:
            reshaped_batches: tuple[torch.Tensor, ...] = reshaped.split(batch_size)

        if conv_args is None:
            return reshaped if batch_size is None else reshaped_batches

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
        return reshaped, conv

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
