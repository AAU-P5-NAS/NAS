import enum
from typing import List, Optional, Tuple
from pydantic import BaseModel, field_validator, model_validator
import torch.nn as nn


class InvalidLayerConfigError(Exception):
    """Raised when a single CNN layer has invalid parameters."""

    pass


class InvalidLayerOrderError(Exception):
    """Raised when CNN layers are in an invalid order (e.g. Conv after Linear)."""

    pass


class CNNExportError(Exception):
    """Raised when CNN layers are in an invalid order (e.g. Conv after Linear)."""

    pass


class StandardAction(enum.Enum):
    REMOVE_LAYER = 0
    MODIFY_LAYER = 1
    ADD_LAYER = 2
    DO_NOTHING = 3


class LayerType(enum.Enum):
    CONV = 0  # "conv"
    LINEAR = 1
    POOL = 2  # "pool"


class LinearUnits(enum.IntEnum):
    LU_8 = 0  # 8
    LU_16 = 1  # 16
    LU_32 = 2  # 32
    LU_64 = 3  # 64
    LU_128 = 4  # 128
    LU_256 = 5  # 256
    LU_512 = 6  # 512

    def to_units(self):
        mapping = [8, 16, 32, 64, 128, 256, 512]
        return mapping[self.value]


class OutChannels(enum.IntEnum):
    CH_16 = 0  # 16
    CH_32 = 1  # 32
    CH_64 = 2  # 64
    CH_128 = 3  # 128

    def to_channels(self):
        mapping = [16, 32, 64, 128]
        return mapping[self.value]


class KernelSize(enum.IntEnum):
    KS_1 = 0  # 1
    KS_3 = 1  # 3
    KS_5 = 2  # 5

    def to_kernel(self):
        mapping = [1, 3, 5]
        return mapping[self.value]


class Stride(enum.IntEnum):
    S_1 = 0  # 1
    S_2 = 1  # 2

    def to_stride(self):
        mapping = [1, 2]
        return mapping[self.value]


class PoolMode(enum.Enum):
    MAX = 0  # "max"
    AVG = 1  # "avg"

    def to_pmode(self):
        mapping = ["max", "avg"]
        return mapping[self.value]


class ActivationFunction(enum.Enum):
    RELU = 0  # "relu"
    TANH = 1  # "tanh"
    SOFTMAX = 2  # "softmax"
    NONE = 3  # "none"

    def to_module(self) -> nn.Module:
        mapping = {
            0: lambda: nn.ReLU(),
            1: lambda: nn.Tanh(),
            2: lambda: nn.Softmax(dim=1),
            3: lambda: nn.Identity(),
        }
        return mapping[self.value]()


class LayerConfig(BaseModel):
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


class NetworkConfig(BaseModel):
    layers: List[LayerConfig]

    @field_validator("layers")
    def check_layer_order(cls, v: List[LayerConfig]) -> List[LayerConfig]:
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
    stride = Stride.to_stride(Stride(stride))
    h_new = (h + 2 * padding - kernel) // stride + 1
    w_new = (w + 2 * padding - kernel) // stride + 1
    return h_new, w_new


def get_latest_layer_index(observation: list[int]):
    """Look for the first occurrence of -1 in the observation array with form index 7, 14, 21 ..."""
    for i in range(0, len(observation), 7):
        if observation[i] == -1 and i != 0:
            return i // 7
        elif observation[i] == -1 and i == 0:
            return None  # No layers defined yet
    return (len(observation) // 7) - 1  # All layers defined


def get_layer_from_index(observation: list[int], index: int) -> LayerConfig:
    """Retrieve the LayerConfig corresponding to a given layer index in the observation."""
    start = index * 7
    if start >= len(observation) or observation[start] == -1:
        raise ValueError(f"Layer index {index} is out of bounds or undefined in the observation.")
    return LayerConfig(
        layer_type=LayerType(observation[start]),
        out_channels=OutChannels(observation[start + 1]) if observation[start + 1] != -1 else None,
        kernel_size=KernelSize(observation[start + 2]) if observation[start + 2] != -1 else None,
        stride=Stride(observation[start + 3]) if observation[start + 3] != -1 else None,
        pool_mode=PoolMode(observation[start + 4]) if observation[start + 4] != -1 else None,
        activation=ActivationFunction(observation[start + 5])
        if observation[start + 5] != -1
        else None,
        linear_units=LinearUnits(observation[start + 6]) if observation[start + 6] != -1 else None,
    )


def get_valid_kernel_sizes(
    last_layer_output_dims: tuple[int, int], padding: int = 1
) -> list[KernelSize]:
    """Get valid kernel sizes based on last layer's output dimensions and padding. Assumes square kernels and no dilation."""
    h, w = last_layer_output_dims
    min_dim = min(h, w)
    max_kernel_size = min_dim + 2 * padding  # ignore stride, chosen later
    valid_kernels = []

    valid_kernels.extend(kernel for kernel in KernelSize if kernel.value <= max_kernel_size)

    return valid_kernels


def calculate_output_dimensions(input_dims: tuple[int, int], layer: LayerConfig) -> tuple[int, int]:
    if layer.kernel_size is None or layer.stride is None:
        return input_dims  # No change if kernel_size or stride is not defined

    h, w = input_dims
    if layer.layer_type == LayerType.CONV:
        h, w = update_spatial_dims(h, w, layer.kernel_size.value, layer.stride.value)
    elif layer.layer_type == LayerType.POOL:
        h, w = update_spatial_dims(h, w, layer.kernel_size.value, layer.stride.value)
    return h, w


def get_output_dimensions(observation: list[int]):
    """Calculate the output dimensions after applying all layers in the observation."""
    input_dims = (28, 28)  # Assuming starting with 28x28 input
    for i in range(0, len(observation), 7):
        if observation[i] == -1:
            break  # No more layers defined
        layer = get_layer_from_index(observation, i // 7)
        input_dims = calculate_output_dimensions(input_dims, layer)
    return input_dims


def get_valid_strides(
    last_layer_output_dims: tuple[int, int], kernel_size: KernelSize, padding: int = 1
) -> list[Stride]:
    """Get valid stride values that won't make output dimensions too small. Assumes square kernels and no dilation."""
    h, w = last_layer_output_dims
    min_dim = min(h, w)
    max_stride = min_dim + 2 * padding - kernel_size.value + 1
    valid_strides = []

    if max_stride < 1:
        return valid_strides

    valid_strides.extend(stride for stride in Stride if stride.value <= max_stride)
    return valid_strides


def get_latest_layer(observation: list[int]):
    """Look for the first occurrence of -1 in the observation array with form index 7, 14, 21 ..."""
    for i in range(0, len(observation), 7):
        if observation[i] == -1 and i != 0:
            return LayerConfig(
                layer_type=LayerType(observation[i]),
                out_channels=OutChannels(observation[i + 1]) if observation[i + 1] != -1 else None,
                kernel_size=KernelSize(observation[i + 2]) if observation[i + 2] != -1 else None,
                stride=Stride(observation[i + 3]) if observation[i + 3] != -1 else None,
                pool_mode=PoolMode(observation[i + 4]) if observation[i + 4] != -1 else None,
                activation=ActivationFunction(observation[i + 5])
                if observation[i + 5] != -1
                else None,
                linear_units=LinearUnits(observation[i + 6]) if observation[i + 6] != -1 else None,
            )
            return i // 7
        elif observation[i] == -1 and i == 0:
            return None  # No layers defined yet
