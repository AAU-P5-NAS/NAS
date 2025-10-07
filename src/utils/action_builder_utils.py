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
    get_latest_layer_index,
    get_output_dimensions,
    get_valid_kernel_sizes,
    get_valid_strides,
)


class MaxLayersReachedException(Exception):
    pass


class ActionStrategy(enum.Enum):
    ADD_LAYER_SEQUENTIAL = "add_layer_sequential"
    ADD_REMOVE_MODIFY = "add_remove_modify"


class SliceWithIndex:
    def __init__(self, slc: slice):
        self._slice = slc

    def get_index(self, enum_value) -> int:
        """Get the absolute index for an enum value within this slice."""
        return self._slice.start + enum_value.value

    @property
    def start(self) -> int:
        return self._slice.start

    @property
    def stop(self) -> int:
        return self._slice.stop

    def __getitem__(self, key):
        """Support creating sub-slices or accessing specific indices."""
        if isinstance(key, slice):
            # Create a new slice within the bounds of this slice
            start = self._slice.start + (key.start or 0)
            stop = min(
                self._slice.start + (key.stop or (self._slice.stop - self._slice.start)),
                self._slice.stop,
            )
            return slice(start, stop, key.step)
        elif isinstance(key, int):
            # Return the absolute index for a relative position
            if key < 0:
                key = (self._slice.stop - self._slice.start) + key
            return self._slice.start + key
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")


class LogitSlices(BaseModel):
    standard_actions: SliceWithIndex
    layer_type: SliceWithIndex
    layer_index: SliceWithIndex
    out_channels: SliceWithIndex
    kernel_size: SliceWithIndex
    stride: SliceWithIndex
    linear_units: SliceWithIndex
    pool_mode: SliceWithIndex
    activation_function: SliceWithIndex
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_index(self, name: str, idx: int) -> int:
        """Safely get an integer index from the named slice."""
        slc = getattr(self, name)
        # For slices, this computes the actual index
        return int(slc.start + idx)


class MaskContext(BaseModel):
    logits: np.ndarray
    observation: list[int]
    slices: LogitSlices
    action_strategy: ActionStrategy
    sampling_strategy: Callable[[np.ndarray], int]
    max_layers: int
    decisions: list[int] = []  # store sampled choice for each head
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
    slices = {}
    for name, size in sizes.items():
        slices[name] = SliceWithIndex(slice(idx, idx + size))
        idx += size
    logit_slices = LogitSlices(**slices)
    return logit_slices


def build_action_add_layer_sequential(ctx: MaskContext):
    ctx.logits = mask_action_type_sequential(ctx)
    ctx.decisions.append(sample_action_from_slice(ctx, "standard_actions"))

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

    return ctx.decisions


def get_logits_for_slice(ctx: MaskContext, slice_name: str) -> np.ndarray:
    slc = getattr(ctx.slices, slice_name)
    return ctx.logits[slc.start : slc.stop]


def sample_action_from_slice(ctx: MaskContext, slice_name: str) -> int:
    logits = get_logits_for_slice(ctx, slice_name)
    choice = ctx.sampling_strategy(logits)
    return choice


def mask_action_type_sequential(ctx: MaskContext):
    """Mask action types based on current observation and strategy. Raises MaxLayersReachedException if max layers reached."""
    new_logits = ctx.logits.copy()

    if get_latest_layer_index(ctx.observation) == ctx.max_layers - 1:
        raise MaxLayersReachedException("Maximum number of layers reached.")

    modify_layer_index = ctx.slices.standard_actions.get_index(StandardAction.MODIFY_LAYER)
    remove_layer_index = ctx.slices.standard_actions.get_index(StandardAction.REMOVE_LAYER)
    new_logits[modify_layer_index] = -np.inf
    new_logits[remove_layer_index] = -np.inf
    return new_logits


def mask_indexes_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()
    latest_layer_index = get_latest_layer_index(ctx.observation)
    if latest_layer_index == ctx.max_layers - 1:
        raise MaxLayersReachedException("Maximum number of layers reached.")
    if latest_layer_index is None:
        latest_layer_index = 0

    layer_index_start = ctx.slices.layer_index.start
    layer_index_end = ctx.slices.layer_index.stop

    new_logits[layer_index_start:layer_index_end] = -np.inf
    new_logits[layer_index_start + latest_layer_index] = 1  # only next index is valid

    return new_logits


def mask_layer_type_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()
    linear_layer_exists = any(
        ctx.observation[i] == LayerType.LINEAR.value for i in range(2, len(ctx.observation), 7)
    )

    if linear_layer_exists:
        layer_type_start = ctx.slices.layer_type.start
        layer_type_end = ctx.slices.layer_type.stop
        new_logits[layer_type_start:layer_type_end] = -np.inf
        linear_index = ctx.slices.layer_type.get_index(LayerType.LINEAR)
        new_logits[linear_index] = 1  # only linear is valid

    return new_logits


def mask_out_channels_sequential(ctx: MaskContext):
    return ctx.logits  # do nothing for now


def mask_kernel_size_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()

    latest_output_dims = get_output_dimensions(ctx.observation)
    valid_kernels = get_valid_kernel_sizes(latest_output_dims)
    invalid_kernels = [k for k in KernelSize if k not in valid_kernels]

    for kernel in invalid_kernels:
        invalid_kernel_index = ctx.slices.kernel_size.get_index(kernel.value)
        new_logits[invalid_kernel_index] = -np.inf

    return new_logits


def mask_stride_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()

    kernel_size_chosen = KernelSize(ctx.decisions[4])

    latest_output_dims = get_output_dimensions(ctx.observation)
    valid_strides = get_valid_strides(latest_output_dims, kernel_size_chosen)
    invalid_strides = [s for s in Stride if s not in valid_strides]

    for stride in invalid_strides:
        invalid_stride_index = ctx.slices.stride.get_index(stride.value)
        new_logits[invalid_stride_index] = -np.inf

    return new_logits


def mask_linear_units_sequential(ctx: MaskContext):
    if ctx.decisions[2] != LayerType.LINEAR.value:
        new_logits = ctx.logits.copy()
        linear_units_start = ctx.slices.linear_units.start
        linear_units_end = ctx.slices.linear_units.stop
        new_logits[linear_units_start:linear_units_end] = -np.inf
        return new_logits

    return ctx.logits


def mask_pool_mode_sequential(ctx: MaskContext):
    if ctx.decisions[2] != LayerType.POOL.value:
        new_logits = ctx.logits.copy()
        pool_mode_start = ctx.slices.pool_mode.start
        pool_mode_end = ctx.slices.pool_mode.stop
        new_logits[pool_mode_start:pool_mode_end] = -np.inf
        return new_logits

    return ctx.logits


def mask_activation_function_sequential(ctx: MaskContext):
    return ctx.logits
