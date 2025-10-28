import enum
from typing import Callable
from pydantic import BaseModel, ConfigDict
import numpy as np

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.utils.network_utils import (
    StandardAction,
    LayerType,
    OutChannels,
    KernelSize,
    Stride,
    LinearUnits,
    PoolMode,
    ActivationFunction,
    get_latest_layer,
    get_latest_layer_index,
    get_output_dimensions,
    get_valid_kernel_sizes,
    get_valid_strides,
)

ACTION_CHOICE = 0
INDEX_CHOICE = 1
LAYER_TYPE_CHOICE = 2
OUT_CHANNELS_CHOICE = 3
KERNEL_SIZE_CHOICE = 4
STRIDE_CHOICE = 5
LINEAR_UNITS_CHOICE = 6
POOL_MODE_CHOICE = 7
ACTIVATION_FUNCTION_CHOICE = 8


class MaxLayersReachedException(Exception):
    """Raised when the maximum number of layers is reached."""

    pass


class ArchitectureCompleteException(Exception):
    """Raised when the architecture is complete and no further actions can be taken."""

    pass


class ActionStrategy(enum.Enum):
    ADD_LAYER_SEQUENTIAL = "add_layer_sequential"
    ADD_REMOVE_MODIFY = "add_remove_modify"


"""
LogitSlice: 
    An enhanced slice wrapper to manage action logits for different action types.
    Provides methods to get absolute indices and access slice properties.

Slices:
    A Pydantic model to hold LogitSlice instances for various action categories.
    Enables structured access to different action slices.

MaskContext:
    A Pydantic model encapsulating the context needed for action masking.
    Contains logits, observation, slices, action strategy, sampling strategy, max layers, and decisions.

The raw logits from the Agent are organized into a Slices object for easier management and sampling. 
This helps with masking specific invalid actions. For example, to mask out linear units: 
    masked_logits = ctx.logits.copy()
    masked_logits[ctx.slices.linear_units.all] = -np.inf # mask all linear units
    masked_logits[ctx.slices.linear_units.idx(LinearUnits.UNIT_128)] = 1  # explicitly allow 128 units
    masked_logits[ctx.slices.linear_units[LinearUnits.UNIT_128]] = -np.inf  # Can also use indexing with [] (same as above)
    return masked_logits 
"""


class LogitSlice:
    def __init__(self, slc: slice):
        self._slice = slc

    def idx(self, enum_value) -> int:
        """Get the absolute index for an enum value within this slice."""
        return self._slice.start + enum_value.value

    @property
    def all(self) -> slice:
        """Property access to the full slice."""
        return self._slice

    @property
    def start(self) -> int:
        """Get the first index of the slice."""
        return self._slice.start

    @property
    def stop(self) -> int:
        """Get the last index of the slice."""
        return self._slice.stop

    def __getitem__(self, key: int | enum.Enum) -> int:
        """Allow [] indexing with enum or int."""
        if isinstance(key, enum.Enum):
            return self.idx(key)
        return self._slice.start + key


class Slices(BaseModel):
    standard_actions: LogitSlice
    layer_type: LogitSlice
    layer_index: LogitSlice
    out_channels: LogitSlice
    kernel_size: LogitSlice
    stride: LogitSlice
    linear_units: LogitSlice
    pool_mode: LogitSlice
    activation_function: LogitSlice
    model_config = ConfigDict(arbitrary_types_allowed=True)


def get_logit_slices(max_layers: int):
    sizes = {
        "standard_actions": len(StandardAction),
        "layer_type": len(LayerType),
        "layer_index": max_layers - 1,
        "out_channels": len(OutChannels),
        "kernel_size": len(KernelSize),
        "stride": len(Stride),
        "linear_units": len(LinearUnits),
        "pool_mode": len(PoolMode),
        "activation_function": len(ActivationFunction),
    }
    idx = 0
    logit_slices = {}
    for name, size in sizes.items():
        logit_slices[name] = LogitSlice(slice(idx, idx + size))
        idx += size
    logit_slices = Slices(**logit_slices)
    return logit_slices


class MaskContext(BaseModel):
    logits: np.ndarray
    observation: np.ndarray
    slices: Slices
    action_strategy: str
    sampling_strategy: Callable[[np.ndarray], int]
    max_layers: int
    decisions: list[int] = []  # store sampled choice for each head
    model_config = ConfigDict(arbitrary_types_allowed=True)


def build_action_add_layer_sequential(ctx: MaskContext):
    try:
        ctx.logits = mask_action_type_sequential(ctx)
        ctx.decisions.append(sample_action_from_slice(ctx, "standard_actions"))
        if ctx.decisions[ACTION_CHOICE] == StandardAction.NONE.value:
            raise ArchitectureCompleteException(
                "Architecture is complete. No further actions can be taken."
            )

        ctx.logits = mask_indexes_sequential(ctx)
        ctx.decisions.append(sample_action_from_slice(ctx, "layer_index"))
        ctx.logits = mask_layer_type_sequential(ctx)
        ctx.decisions.append(sample_action_from_slice(ctx, "layer_type"))
        ctx.logits = mask_out_channels_sequential(ctx)
        ctx.decisions.append(sample_action_from_slice(ctx, "out_channels"))
        ctx.logits = mask_kernel_size_sequential(ctx)
        ctx.decisions.append(sample_action_from_slice(ctx, "kernel_size"))
        ctx.logits = mask_stride_sequential(ctx)
        ctx.decisions.append(sample_action_from_slice(ctx, "stride"))
        ctx.logits = mask_linear_units_sequential(ctx)
        ctx.decisions.append(sample_action_from_slice(ctx, "linear_units"))
        ctx.logits = mask_pool_mode_sequential(ctx)
        ctx.decisions.append(sample_action_from_slice(ctx, "pool_mode"))
        ctx.logits = mask_activation_function_sequential(ctx)
        ctx.decisions.append(sample_action_from_slice(ctx, "activation_function"))

    except Exception:
        return None  # no action to perform.

    return ctx.decisions


def sample_action_from_slice(ctx: MaskContext, slice_name: str) -> int:
    logits = ctx.logits[getattr(ctx.slices, slice_name).all]
    if np.all(logits == -np.inf):
        choice = 0  # No valid actions available
    else:
        choice = int(ctx.sampling_strategy(logits))
    return choice


def mask_action_type_sequential(ctx: MaskContext):
    """Mask action types based on current observation and strategy. Raises MaxLayersReachedException if max layers reached."""
    new_logits = ctx.logits.copy()

    latest_layer_index = get_latest_layer_index(ctx.observation)
    if latest_layer_index == ctx.max_layers - 1:
        raise MaxLayersReachedException("Maximum number of layers reached.")

    if latest_layer_index is None:
        # No layers yet, can only add
        new_logits[ctx.slices.standard_actions.all] = -np.inf
        new_logits[ctx.slices.standard_actions[StandardAction.ADD_LAYER]] = 1
        return new_logits

    modify_layer_index = ctx.slices.standard_actions[StandardAction.MODIFY_LAYER]
    remove_layer_index = ctx.slices.standard_actions[StandardAction.REMOVE_LAYER]
    new_logits[modify_layer_index] = -np.inf
    new_logits[remove_layer_index] = -np.inf
    return new_logits


def mask_indexes_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()

    latest_layer_index = get_latest_layer_index(ctx.observation)
    if latest_layer_index == ctx.max_layers - 1:
        raise MaxLayersReachedException("Maximum number of layers reached.")

    next_layer_index = latest_layer_index + 1 if latest_layer_index is not None else 0
    new_logits[ctx.slices.layer_index.all] = -np.inf

    layer_index_start = ctx.slices.layer_index.start
    new_logits[layer_index_start + next_layer_index] = 1  # only next index is valid

    return new_logits


def mask_layer_type_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()

    if ctx.decisions[ACTION_CHOICE] == StandardAction.NONE.value:
        new_logits[ctx.slices.layer_type.all] = -np.inf
        new_logits[ctx.slices.layer_type[LayerType.NONE]] = 1
        return new_logits

    linear_layer_exists = any(
        ctx.observation[i] == LayerType.LINEAR.value for i in range(0, len(ctx.observation), 7)
    )
    if linear_layer_exists:
        new_logits[ctx.slices.layer_type.all] = -np.inf
        new_logits[ctx.slices.layer_type[LayerType.LINEAR]] = 1  # only linear is valid

    previous_layer = get_latest_layer(ctx.observation)
    if previous_layer is None or previous_layer.layer_type != LayerType.CONV:
        # if no previous layer or previous layer is not conv, cannot add pool
        new_logits[ctx.slices.layer_type[LayerType.POOL]] = -np.inf

    new_logits[
        ctx.slices.layer_type[LayerType.NONE]
    ] = -np.inf  # NONE is not valid when adding a layer
    return new_logits


def mask_out_channels_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()

    if (
        ctx.decisions[LAYER_TYPE_CHOICE] == LayerType.LINEAR.value
        or ctx.decisions[LAYER_TYPE_CHOICE] == LayerType.NONE.value
    ):
        new_logits[ctx.slices.out_channels.all] = -np.inf
        new_logits[ctx.slices.out_channels[OutChannels.NONE]] = 1
        return new_logits

    new_logits[ctx.slices.out_channels[OutChannels.NONE]] = -np.inf  # NONE is not valid when adding
    return new_logits


def mask_kernel_size_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()

    if (
        ctx.decisions[LAYER_TYPE_CHOICE] == LayerType.NONE.value
        or ctx.decisions[LAYER_TYPE_CHOICE] == LayerType.LINEAR.value
    ):
        # mask all kernel sizes
        new_logits[ctx.slices.kernel_size.all] = -np.inf
        new_logits[ctx.slices.kernel_size[KernelSize.NONE]] = 1
        return new_logits

    latest_output_dims = get_output_dimensions(ctx.observation)
    valid_kernels = get_valid_kernel_sizes(latest_output_dims)
    invalid_kernels = [k for k in KernelSize if k not in valid_kernels]

    for kernel in invalid_kernels:
        invalid_kernel_index = ctx.slices.kernel_size[kernel.value]
        new_logits[invalid_kernel_index] = -np.inf

    new_logits[ctx.slices.kernel_size[KernelSize.NONE]] = -np.inf
    return new_logits


def mask_stride_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()

    if (
        ctx.decisions[LAYER_TYPE_CHOICE] == LayerType.NONE.value
        or ctx.decisions[LAYER_TYPE_CHOICE] == LayerType.LINEAR.value
    ):
        # mask all strides
        new_logits[ctx.slices.stride.all] = -np.inf
        new_logits[ctx.slices.stride[Stride.NONE]] = 1
        return new_logits

    kernel_size_chosen = KernelSize(ctx.decisions[KERNEL_SIZE_CHOICE])
    latest_output_dims = get_output_dimensions(ctx.observation)
    valid_strides = get_valid_strides(latest_output_dims, kernel_size_chosen)
    invalid_strides = [s for s in Stride if s not in valid_strides]

    for stride in invalid_strides:
        invalid_stride_index = ctx.slices.stride[stride.value]
        new_logits[invalid_stride_index] = -np.inf

    new_logits[ctx.slices.stride[Stride.NONE]] = -np.inf
    return new_logits


def mask_linear_units_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()

    if ctx.decisions[LAYER_TYPE_CHOICE] != LayerType.LINEAR.value:
        new_logits[ctx.slices.linear_units.all] = -np.inf
        new_logits[ctx.slices.linear_units[LinearUnits.NONE]] = 1
        return new_logits

    new_logits[ctx.slices.linear_units[LinearUnits.NONE]] = -np.inf
    return new_logits


def mask_pool_mode_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()

    if ctx.decisions[LAYER_TYPE_CHOICE] != LayerType.POOL.value:
        new_logits[ctx.slices.pool_mode.all] = -np.inf
        new_logits[ctx.slices.pool_mode[PoolMode.NONE]] = 1
        return new_logits

    return ctx.logits


def mask_activation_function_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()
    # Mask NONE activation - not useful for hidden layers
    new_logits[ctx.slices.activation_function[ActivationFunction.NONE]] = -np.inf
    # Mask SOFTMAX activation - should only be used in output layer, not hidden layers
    new_logits[ctx.slices.activation_function[ActivationFunction.SOFTMAX]] = -np.inf
    return new_logits
