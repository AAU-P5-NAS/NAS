import torch.nn as nn
import torch.onnx
import onnx
import os
import warnings
import enum
from enum import IntEnum
from typing import List, Optional, Tuple
from pydantic import BaseModel, field_validator, model_validator


class InvalidLayerConfigError(Exception):
    """Raised when a single CNN layer has invalid parameters."""

    pass


class InvalidLayerOrderError(Exception):
    """Raised when CNN layers are in an invalid order (e.g. Conv after Linear)."""

    pass


class CNNExportError(Exception):
    """Raised when CNN layers are in an invalid order (e.g. Conv after Linear)."""

    pass


class LayerType(enum.Enum):
    CONV = "conv"
    LINEAR = "linear"
    POOL = "pool"


class LinearUnits(IntEnum):
    LU_64 = 64
    LU_128 = 128
    LU_256 = 256
    LU_512 = 512


class OutChannels(IntEnum):
    CH_16 = 16
    CH_32 = 32
    CH_64 = 64
    CH_128 = 128


class KernelSize(IntEnum):
    KS_1 = 1
    KS_3 = 3
    KS_5 = 5


class Stride(IntEnum):
    S_1 = 1
    S_2 = 2


class PoolMode(enum.Enum):
    MAX = "max"
    AVG = "avg"


class ActivationFunction(enum.Enum):
    RELU = "relu"
    TANH = "tanh"
    SOFTMAX = "softmax"
    NONE = "none"

    def to_module(self) -> nn.Module:
        """Map enum -> actual nn.Module instance."""
        mapping = {
            "relu": lambda: nn.ReLU(),
            "tanh": lambda: nn.Tanh(),
            "softmax": lambda: nn.Softmax(dim=1),
            "none": lambda: nn.Identity(),
        }
        return mapping[self.value]()


class CNNActionSpace(BaseModel):
    layer_type: LayerType
    out_channels: Optional[OutChannels] = None
    kernel_size: Optional[KernelSize] = None
    stride: Optional[Stride] = None
    pool_mode: Optional[PoolMode] = None
    activation: Optional[ActivationFunction] = ActivationFunction.NONE
    linear_units: Optional[LinearUnits] = None

    @model_validator(mode="after")
    def validate_params(self):
        lt = self.layer_type
        if lt == LayerType.CONV:
            if self.out_channels is None or self.kernel_size is None:
                raise InvalidLayerConfigError("Conv layer must define out_channels and kernel_size")
        elif lt == LayerType.POOL:
            if self.pool_mode is None or self.kernel_size is None:
                raise InvalidLayerConfigError("Pool layer must define pool_mode and kernel_size")
        elif lt == LayerType.LINEAR:
            if self.linear_units is None:
                raise InvalidLayerConfigError("Linear layer must define linear_units")
        return self


class RLConfig(BaseModel):
    layers: List[CNNActionSpace]

    @field_validator("layers")
    def check_layer_order(cls, v: List[CNNActionSpace]) -> List[CNNActionSpace]:
        """Enforce conv/pool layers cannot appear after a linear layer."""
        seen_linear = False
        for i, layer in enumerate(v):
            if seen_linear and layer.layer_type in (LayerType.CONV, LayerType.POOL):
                raise InvalidLayerOrderError(
                    f"Conv/Pool layer at position {i} after a linear layer is not allowed"
                )
            if layer.layer_type == LayerType.LINEAR:
                seen_linear = True
        return v


def update_spatial_dims(
    h: int, w: int, kernel: int, stride: int, padding: int = 0
) -> Tuple[int, int]:
    h_new = (h + 2 * padding - kernel) // stride + 1
    w_new = (w + 2 * padding - kernel) // stride + 1
    return h_new, w_new


class CNNBuilder:
    def __init__(
        self, rl_config: RLConfig, input_size: Tuple[int, int] = (28, 28), num_classes: int = 26
    ):
        self.rl_config = rl_config
        self.input_size = input_size
        self.num_classes = num_classes
        self.model: Optional[nn.Sequential] = None

    def build(self):
        layers = []
        current_in_channels = 1
        h, w = self.input_size

        conv_pool_layers = [
            layer
            for layer in self.rl_config.layers
            if layer.layer_type in (LayerType.CONV, LayerType.POOL)
        ]
        linear_layers = [
            layer for layer in self.rl_config.layers if layer.layer_type is LayerType.LINEAR
        ]

        for layer in conv_pool_layers:
            if layer.layer_type is LayerType.CONV:
                stride = layer.stride or 1
                kernel = layer.kernel_size
                assert kernel is not None  #
                padding = kernel // 2

                out_ch = layer.out_channels
                assert out_ch is not None
                assert current_in_channels is not None

                layers.append(
                    nn.Conv2d(
                        in_channels=current_in_channels,
                        out_channels=out_ch,
                        kernel_size=kernel,
                        stride=stride,
                        padding=padding,
                    )
                )
                assert layer.activation is not None
                layers.append(layer.activation.to_module())
                current_in_channels = layer.out_channels
                h, w = update_spatial_dims(h, w, kernel, stride, padding)

            elif layer.layer_type is LayerType.POOL:
                stride = layer.stride or layer.kernel_size
                kernel = layer.kernel_size
                assert kernel is not None
                if layer.pool_mode is PoolMode.MAX:
                    layers.append(nn.MaxPool2d(kernel, stride))
                else:
                    layers.append(nn.AvgPool2d(kernel, stride))

                assert stride is not None
                h, w = update_spatial_dims(h, w, kernel, stride)

        layers.append(nn.Flatten())
        assert current_in_channels is not None

        in_features = current_in_channels * h * w

        for layer in linear_layers:
            assert in_features is not None
            assert layer.linear_units is not None
            layers.append(nn.Linear(in_features, layer.linear_units))
            assert layer.activation is not None
            layers.append(layer.activation.to_module())
            in_features = layer.linear_units

        layers.append(nn.Linear(in_features, self.num_classes))

        self.model = nn.Sequential(*layers)
        return self.model

    def export_to_onnx(self, input_size=(1, 28, 28), filename=None, opset=17):
        """
        Export the built CNN to ONNX format and save it a seperate file.

        Note, ONNX just mirrors the PyTorch model at the time of export.

        Args:
            input_size: tuple (C, H, W) of the input image (excluding batch size)
            filename: ONNX file name
            opset: ONNX opset version
        """
        if self.model is None:
            raise CNNExportError("Build the model first with .build() before exporting.")

        if not isinstance(input_size, tuple):
            raise CNNExportError("input_size must be a tuple (C, H, W)")

        if filename is None:
            # Auto-generate filename
            filename = f"cnn_model_{id(self)}.onnx"

        os.makedirs("saved_models", exist_ok=True)
        full_path = os.path.join("saved_models", filename)

        dummy_input = torch.randn(1, *input_size)
        # Suppress deprecation warnings for ONNX export. smthn about a new version of onnx exporter, but the current one still work
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            torch.onnx.export(
                self.model,
                (dummy_input,),  # tuple of inputs
                full_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                opset_version=opset,
                export_params=True,
                training=torch.onnx.TrainingMode.EVAL,  # export in inference mode
            )

        # Verify ONNX model
        onnx_model = onnx.load(full_path)
        onnx.checker.check_model(onnx_model)
        return full_path


if __name__ == "__main__":
    # Define a sample RL-generated config
    config = RLConfig(
        layers=[
            CNNActionSpace(
                layer_type=LayerType.CONV,
                out_channels=OutChannels.CH_16,
                kernel_size=KernelSize.KS_3,
                activation=ActivationFunction.RELU,
            ),
            CNNActionSpace(
                layer_type=LayerType.POOL, pool_mode=PoolMode.MAX, kernel_size=KernelSize.KS_1
            ),
            CNNActionSpace(
                layer_type=LayerType.LINEAR,
                linear_units=LinearUnits.LU_64,
                activation=ActivationFunction.TANH,
            ),
        ]
    )

    # Instantiate the CNN builder
    cnn_builder = CNNBuilder(rl_config=config, input_size=(28, 28), num_classes=26)

    # Build the PyTorch model
    model = cnn_builder.build()

    # Print the model architecture
    print("Built CNN model:")
    print(model)

    # Optional: export to ONNX
    # onnx_path = cnn_builder.export_to_onnx(input_size=(1, 28, 28))
    # print(f"ONNX model saved at: {onnx_path}")
