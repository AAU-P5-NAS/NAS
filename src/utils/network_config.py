from typing import List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, field_validator
from src.agent.action_masking.action_masking_utils import MaskContext
from src.utils.layer_config import LayerConfig, LayerType, OutChannels, KernelSize, Stride, LinearUnits, PoolMode, ActivationFunction

class InvalidLayerConfigError(Exception):
    """Raised when a single CNN layer has invalid parameters."""
    pass


class InvalidLayerOrderError(Exception):
    """Raised when CNN layers are in an invalid order (e.g. Conv after Linear)."""
    pass


 # format [layer_type, out_channels, kernel_size, stride, pool_mode, activation, linear_units, skip_connection]
SINGLE_LAYER_OBSERVATION_SIZE = 8 

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
    from src.utils.architecture import flatten_cnn_config

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
