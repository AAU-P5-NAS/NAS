import enum
from typing import List, Optional, Tuple, TYPE_CHECKING
import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
import torch.nn as nn

if TYPE_CHECKING:
    # Only imported for type checking to avoid circular imports at runtime.
    from src.agent.action_masking.action_masking_utils import MaskContext


class InvalidLayerConfigError(Exception):
    """Raised when a single CNN layer has invalid parameters."""
    pass


class InvalidLayerOrderError(Exception):
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

    def to_module(self) -> nn.Module:
        mapping = {
            0: lambda: nn.Identity(),
            1: lambda: nn.ReLU(),
        }
        return mapping[self.value]()

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

class LayerConfig(BaseModel):
    layer_type: LayerType
    out_channels: OutChannels = OutChannels.NONE
    kernel_size: KernelSize = KernelSize.NONE
    stride: Stride = Stride.NONE
    pool_mode: PoolMode = PoolMode.NONE
    activation: ActivationFunction = ActivationFunction.NONE
    linear_units: LinearUnits = LinearUnits.NONE
    skip_connection: Optional[int] = None  # index of the layer to skip from

    @classmethod
    def from_latest_observation(cls, observation: np.ndarray):
        layer_type = LayerType(int(observation[0])) if observation[0] != 0 else LayerType.NONE
        out_channels = OutChannels(int(observation[1])) if observation[1] != 0 else OutChannels.NONE
        kernel_size = KernelSize(int(observation[2])) if observation[2] != 0 else KernelSize.NONE
        stride = Stride(int(observation[3])) if observation[3] != 0 else Stride.NONE
        pool_mode = PoolMode(int(observation[4])) if observation[4] != 0 else PoolMode.NONE
        activation = ActivationFunction(int(observation[5])) if observation[5] != 0 else ActivationFunction.NONE
        linear_units = LinearUnits(int(observation[6])) if observation[6] != 0 else LinearUnits.NONE
        skip_connection = int(observation[7]) if observation[7] != 0 else None

        return cls(
            layer_type=layer_type,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pool_mode=pool_mode,
            activation=activation,
            linear_units=linear_units,
            skip_connection=skip_connection,
        )
    
    @classmethod
    def from_decisions(cls, actions:Decisions):
        layer_type = LayerType(actions.layer_type_choice)
        out_channels = OutChannels(actions.out_channels_choice)
        kernel_size = KernelSize(actions.kernel_size_choice)
        stride = Stride(actions.stride_choice)
        linear_units = LinearUnits(actions.linear_units_choice)
        pool_mode = PoolMode(actions.pool_mode_choice)
        activation = ActivationFunction(actions.activation_function_choice)
        skip_connection = actions.skip_connection_choice

        return cls(
            layer_type=layer_type,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pool_mode=pool_mode,
            activation=activation,
            linear_units=linear_units,
            skip_connection=skip_connection,
        )

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

class NetworkConfig(BaseModel):
    layers: List[LayerConfig]
    def __add__(self, other: LayerConfig) -> "NetworkConfig":
    
        # add_layer mutates and returns the partial architecture
        self.layers.append(other)
        return self

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
    
def transform_action_indices_to_decisions(action_indices: np.ndarray, max_layers:int):
    actions_as_ints = [int(x) for x in np.array(action_indices).flatten()]

    if len(actions_as_ints) < 9:
        raise ValueError(f"Expected action vector with >=9 entries, got {len(actions_as_ints)}: {action_indices}")

    return Decisions(
        action_choice=StandardAction(actions_as_ints[0]),
        layer_type_choice=LayerType(actions_as_ints[1]),
        out_channels_choice=OutChannels(actions_as_ints[2]),
        kernel_size_choice=KernelSize(actions_as_ints[3]),
        stride_choice=Stride(actions_as_ints[4]),
        linear_units_choice=LinearUnits(actions_as_ints[5]),
        pool_mode_choice=PoolMode(actions_as_ints[6]),
        activation_function_choice=ActivationFunction(actions_as_ints[7]),
        skip_connection_choice=actions_as_ints[8] if actions_as_ints[8] != max_layers - 1 else None 
    )

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
    return LayerConfig.from_latest_observation(observation[start:start+SINGLE_LAYER_OBSERVATION_SIZE])

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
        print("hw: ",h, w)
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
        print("LAYER", layer)
        input_dims = calculate_output_dimensions(input_dims, layer)
        print("outdim", input_dims[0], " ," ,input_dims[1])
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

def get_observation_with_new_layer(observation, ctx: "MaskContext"):
    # Local import to avoid circular import at module import time.
    from src.utils.graph_cnn import flatten_cnn_config

    observation = observation.copy()
    new_layer = LayerConfig.from_decisions(ctx.decisions)
    new_layer_flattened = np.array(
        flatten_cnn_config(NetworkConfig(layers=[new_layer]), ctx.max_layers, padded_with_zeros=False),
        dtype=np.float32,
    )
    if len(new_layer_flattened) != SINGLE_LAYER_OBSERVATION_SIZE:
        raise Exception("incorrect layer size")
    last_layer_index = ctx.action_count
    observation[last_layer_index*SINGLE_LAYER_OBSERVATION_SIZE:(last_layer_index+1)*SINGLE_LAYER_OBSERVATION_SIZE] = new_layer_flattened
    
    return observation

def get_latest_layer(
    observation: np.ndarray, action_count: int, max_layers: int
) -> Optional[LayerConfig]:
    # action_count = number of layers
    if action_count == 0:
        return None 

    idx = (action_count - 1) * SINGLE_LAYER_OBSERVATION_SIZE
    layer_type = LayerType(int(observation[idx])) if observation[idx] != 0 else LayerType.NONE
    out_channels = OutChannels(int(observation[idx + 1])) if observation[idx + 1] != 0 else OutChannels.NONE
    kernel_size = KernelSize(int(observation[idx + 2])) if observation[idx + 2] != 0 else KernelSize.NONE
    stride = Stride(int(observation[idx + 3])) if observation[idx + 3] != 0 else Stride.NONE
    pool_mode = PoolMode(int(observation[idx + 4])) if observation[idx + 4] != 0 else PoolMode.NONE
    activation = ActivationFunction(int(observation[idx + 5])) if observation[idx + 5] != 0 else ActivationFunction.NONE
    linear_units = LinearUnits(int(observation[idx + 6])) if observation[idx + 6] != 0 else LinearUnits.NONE
    skip_connection = int(observation[idx + 7]) if observation[idx + 7] != max_layers - 1 else None

    return LayerConfig(
        layer_type=layer_type,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        pool_mode=pool_mode,
        activation=activation,
        linear_units=linear_units,
        skip_connection=skip_connection,
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
