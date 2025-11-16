import enum
from typing import List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
import torch.nn as nn


class InvalidActionError(Exception):
    """Raised when an invalid action is provided to modify the network architecture."""

    pass


class InvalidLayerConfigError(Exception):
    """Raised when a single CNN layer has invalid parameters."""

    pass


class InvalidLayerOrderError(Exception):
    """Raised when CNN layers are in an invalid order (e.g. Conv after Linear)."""

    pass


class CNNExportError(Exception):
    """Raised when CNN layers are in an invalid order (e.g. Conv after Linear)."""

    pass


SINGLE_LAYER_OBSERVATION_SIZE = 8  # format [layer_type, out_channels, kernel_size, stride, pool_mode, activation, linear_units, skip_connection]


class StandardAction(enum.Enum):
    NONE = 0
    ADD_LAYER = 1


class LayerType(enum.IntEnum):
    NONE = 0
    CONV = 1  # "conv"
    LINEAR = 2
    POOL = 3  # "pool"


class LinearUnits(enum.IntEnum):
    NONE = 0
    LU_64 = 1  # 64
    LU_128 = 2  # 128
    LU_256 = 3  # 256
    LU_512 = 4  # 512

    def to_units(self):
        mapping = [None, 64, 128, 256, 512]
        return mapping[self.value]


class OutChannels(enum.IntEnum):
    NONE = 0
    CH_16 = 1  # 16
    CH_32 = 2  # 32
    CH_64 = 3  # 64
    CH_128 = 4  # 128
    CH_256 = 5  # 256

    def to_channels(self):
        mapping = [None, 16, 32, 64, 128, 256]
        return mapping[self.value]


class KernelSize(enum.IntEnum):
    NONE = 0
    KS_1 = 1  # 1
    KS_3 = 2  # 3
    KS_5 = 3  # 5

    def to_kernel(self):
        mapping = [None, 1, 3, 5]
        return mapping[self.value]


class Stride(enum.IntEnum):
    NONE = 0
    S_1 = 1  # 1
    S_2 = 2  # 2

    def to_stride(self):
        mapping = [None, 1, 2]
        return mapping[self.value]


class PoolMode(enum.Enum):
    NONE = 0
    MAX = 1  # "max"
    AVG = 2  # "avg"

    def to_pmode(self):
        mapping = [None, "max", "avg"]
        return mapping[self.value]


class ActivationFunction(enum.Enum):
    NONE = 0  # "none"
    RELU = 1  # "relu"
    TANH = 2  # "tanh"
    SOFTMAX = 3  # "softmax"

    def to_module(self) -> nn.Module:
        mapping = {
            0: lambda: nn.Identity(),
            1: lambda: nn.ReLU(),
            2: lambda: nn.Tanh(),
            3: lambda: nn.Softmax(dim=1),
        }
        return mapping[self.value]()


class LayerConfig(BaseModel):
    layer_type: LayerType
    out_channels: OutChannels = OutChannels.NONE
    kernel_size: KernelSize = KernelSize.NONE
    stride: Stride = Stride.NONE
    pool_mode: PoolMode = PoolMode.NONE
    activation: ActivationFunction = ActivationFunction.NONE
    linear_units: LinearUnits = LinearUnits.NONE
    skip_connection: Optional[int] = None  # index of the layer to skip from

    @model_validator(mode="after")
    def validate_params(self):
        lt = self.layer_type
        if lt == LayerType.CONV:
            if self.out_channels == OutChannels.NONE or self.kernel_size == KernelSize.NONE:
                raise InvalidLayerConfigError("Conv layer must define out_channels and kernel_size")
        elif lt == LayerType.POOL:
            if self.pool_mode == PoolMode.NONE or self.kernel_size == KernelSize.NONE:
                raise InvalidLayerConfigError("Pool layer must define pool_mode and kernel_size")
        elif lt == LayerType.LINEAR:
            if self.linear_units == LinearUnits.NONE:
                raise InvalidLayerConfigError("Linear layer must define linear_units")
        return self


class Decisions(BaseModel):
    action_choice: StandardAction
    layer_type_choice: LayerType
    out_channels_choice: OutChannels
    kernel_size_choice: KernelSize
    stride_choice: Stride
    linear_units_choice: LinearUnits
    pool_mode_choice: PoolMode
    activation_function_choice: ActivationFunction
    skip_connection_choice: Optional[int]
    model_config = ConfigDict(arbitrary_types_allowed=True)


EMPTY_DECISIONS = Decisions(
    action_choice=StandardAction.NONE,
    layer_type_choice=LayerType.NONE,
    out_channels_choice=OutChannels.NONE,
    kernel_size_choice=KernelSize.NONE,
    stride_choice=Stride.NONE,
    linear_units_choice=LinearUnits.NONE,
    pool_mode_choice=PoolMode.NONE,
    skip_connection_choice=None,
    activation_function_choice=ActivationFunction.NONE,
)


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

    def extend(self, action: Decisions, partial_arch: "NetworkConfig"):
        """
        Input: Takes a list of action and partially builds architecture

        Output: Returns the partially built architecture with the new layer appended to it

        Note:
        The actions must be in the following order, otherwise, the method will fail when calling .build() on the constructed Network:
        [action, layerType, outCh, kernelSize, stride, linearU, poolMode, actFun]

        Currently, it only appends the layer at the end.
        """
        if action.action_choice == StandardAction.NONE:
            return partial_arch

        return self.add_layer(action, partial_arch)

    def add_layer(self, actions: Decisions, partial_arch: "NetworkConfig") -> "NetworkConfig":
        lt = LayerType(actions.layer_type_choice)
        oc = OutChannels(actions.out_channels_choice)
        ks = KernelSize(actions.kernel_size_choice)
        st = Stride(actions.stride_choice)
        lu = LinearUnits(actions.linear_units_choice)
        pm = PoolMode(actions.pool_mode_choice)
        act = ActivationFunction(actions.activation_function_choice)

        layer_config = LayerConfig(
            layer_type=lt,
            out_channels=oc,
            kernel_size=ks,
            stride=st,
            linear_units=lu,
            pool_mode=pm,
            activation=act,
        )

        partial_arch.layers.append(layer_config)

        return partial_arch


def get_number_of_actions_from_observation(observation: np.ndarray) -> int:
    """Count the number of defined layers in the observation array."""
    count = 0
    for i in range(0, len(observation), SINGLE_LAYER_OBSERVATION_SIZE):
        if observation[i] == 0:
            break
        count += 1
    return count


def update_spatial_dims(
    h: int, w: int, kernel: int, stride: int, padding: int = 0
) -> Tuple[int, int]:
    h_new = (h + 2 * padding - kernel) // stride + 1
    w_new = (w + 2 * padding - kernel) // stride + 1
    return h_new, w_new


'''
def get_latest_layer_index(observation: np.ndarray):
    """Look for the first occurrence of 0 in the observation array with form index 8, 16, 24 ..."""
    for i in range(0, len(observation), SINGLE_LAYER_OBSERVATION_SIZE):
        if observation[i] == 0 and i != 0:
            return i // SINGLE_LAYER_OBSERVATION_SIZE - 1
        elif observation[i] == 0 and i == 0:
            return None  # No layers defined yet
    return (len(observation) // SINGLE_LAYER_OBSERVATION_SIZE) - 1  # All layers defined
'''


def get_layer_from_index(observation: np.ndarray, index: int, max_layers: int) -> LayerConfig:
    """Retrieve the LayerConfig corresponding to a given layer index in the observation."""
    start = index * SINGLE_LAYER_OBSERVATION_SIZE
    if start >= len(observation) or observation[start] == -1:
        raise ValueError(f"Layer index {index} is out of bounds or undefined in the observation.")
    return LayerConfig(
        layer_type=LayerType(observation[start]),
        out_channels=OutChannels(observation[start + 1])
        if observation[start + 1] != 0
        else OutChannels.NONE,
        kernel_size=KernelSize(observation[start + 2])
        if observation[start + 2] != 0
        else KernelSize.NONE,
        stride=Stride(observation[start + 3]) if observation[start + 3] != 0 else Stride.NONE,
        pool_mode=PoolMode(observation[start + 4])
        if observation[start + 4] != 0
        else PoolMode.NONE,
        activation=ActivationFunction(observation[start + 5])
        if observation[start + 5] != 0
        else ActivationFunction.NONE,
        linear_units=LinearUnits(observation[start + 6])
        if observation[start + 6] != 0
        else LinearUnits.NONE,
        skip_connection=observation[start + 7]
        if observation[start + 7] != max_layers - 1
        else None,
    )


def get_valid_kernel_sizes(
    last_layer_output_dims: tuple[int, int], padding: int = 0
) -> list[KernelSize]:
    h, w = last_layer_output_dims
    min_dim = min(h, w)
    max_kernel_size = min_dim  # ignore padding entirely
    valid_kernels = [
        kernel
        for kernel in KernelSize
        if kernel.to_kernel() is not None and kernel.to_kernel() <= max_kernel_size
    ]
    return valid_kernels


def calculate_output_dimensions(input_dims: tuple[int, int], layer: LayerConfig) -> tuple[int, int]:
    if layer.kernel_size is None or layer.stride is None:
        return input_dims  # No change if kernel_size or stride is not defined

    h, w = input_dims
    if layer.layer_type == LayerType.CONV:
        padding = layer.kernel_size.to_kernel() // 2  # assuming same padding
        h, w = update_spatial_dims(
            h, w, layer.kernel_size.to_kernel(), layer.stride.to_stride(), padding
        )
    elif layer.layer_type == LayerType.POOL:
        h, w = update_spatial_dims(h, w, layer.kernel_size.to_kernel(), layer.stride.to_stride())
    return h, w


def get_output_dimensions(
    observation: np.ndarray, input_dims: tuple[int, int], max_layers: int
) -> tuple[int, int]:
    """Calculate the output dimensions after applying all layers in the observation."""
    for i in range(0, len(observation), SINGLE_LAYER_OBSERVATION_SIZE):
        if observation[i] == 0:
            break  # No more layers defined
        layer = get_layer_from_index(observation, i // SINGLE_LAYER_OBSERVATION_SIZE, max_layers)
        input_dims = calculate_output_dimensions(input_dims, layer)
    return input_dims


def get_valid_strides(
    last_layer_output_dims: tuple[int, int], kernel_size: KernelSize, padding: int = 1
) -> list[Stride]:
    """Get valid stride values that won't make output dimensions too small. Assumes square kernels and no dilation."""
    h, w = last_layer_output_dims
    min_dim = min(h, w)
    max_stride = min_dim + 2 * padding - kernel_size.to_kernel() + 1
    valid_strides = []

    if max_stride < 1:
        return valid_strides

    valid_strides.extend(
        stride
        for stride in Stride
        if stride.to_stride() is not None and stride.to_stride() <= max_stride
    )
    return valid_strides


def get_latest_layer(
    observation: np.ndarray, action_count: int, max_layers: int
) -> Optional[LayerConfig]:
    # action_count = number of layers
    if action_count == 0:
        return None 

    last_layer_index = action_count - 1
    idx = last_layer_index * SINGLE_LAYER_OBSERVATION_SIZE
    return LayerConfig(
        layer_type=LayerType(observation[idx]),
        out_channels=OutChannels(observation[idx + 1])
        if observation[idx + 1] != 0
        else OutChannels.NONE,
        kernel_size=KernelSize(observation[idx + 2])
        if observation[idx + 2] != 0
        else KernelSize.NONE,
        stride=Stride(observation[idx + 3]) if observation[idx + 3] != 0 else Stride.NONE,
        pool_mode=PoolMode(observation[idx + 4]) if observation[idx + 4] != 0 else PoolMode.NONE,
        activation=ActivationFunction(observation[idx + 5])
        if observation[idx + 5] != 0
        else ActivationFunction.NONE,
        linear_units=LinearUnits(observation[idx + 6])
        if observation[idx + 6] != 0
        else LinearUnits.NONE,
        skip_connection=observation[idx + 7] if observation[idx + 7] != max_layers - 1 else None,
    )

    """for i in range(0, len(observation), SINGLE_LAYER_OBSERVATION_SIZE):
        if observation[i] == 0 and i != 0:
            idx = i - SINGLE_LAYER_OBSERVATION_SIZE
            last_layer_index = idx // SINGLE_LAYER_OBSERVATION_SIZE - 1
            return LayerConfig(
                layer_type=LayerType(observation[idx]),
                out_channels=OutChannels(observation[idx + 1])
                if observation[idx + 1] != 0
                else OutChannels.NONE,
                kernel_size=KernelSize(observation[idx + 2])
                if observation[idx + 2] != 0
                else KernelSize.NONE,
                stride=Stride(observation[idx + 3]) if observation[idx + 3] != 0 else Stride.NONE,
                pool_mode=PoolMode(observation[idx + 4])
                if observation[idx + 4] != 0
                else PoolMode.NONE,
                activation=ActivationFunction(observation[idx + 5])
                if observation[idx + 5] != 0
                else ActivationFunction.NONE,
                linear_units=LinearUnits(observation[idx + 6])
                if observation[idx + 6] != 0
                else LinearUnits.NONE,
                skip_connection=observation[idx + 7]
                if observation[idx + 7]
                != (last_layer_index)  # current layer index => no skip connection
                else -1,
            )
        elif observation[i] == 0 and i == 0:
            return None  # No layers defined yet
            """
