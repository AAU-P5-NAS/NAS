import enum
import numpy as np
import sys
import os
from typing import Callable
from pydantic import BaseModel, ConfigDict

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


class MaxLayersReachedException(Exception):
    """Raised when the maximum number of layers is reached."""

    pass


class ActionStrategy(enum.Enum):
    ADD_LAYER_SEQUENTIAL = "add_layer_sequential"
    ADD_REMOVE_MODIFY = "add_remove_modify"


class LogitSlices(BaseModel):
    standard_actions: slice
    layer_type: slice
    layer_index: slice
    out_channels: slice
    kernel_size: slice
    stride: slice
    linear_units: slice
    pool_mode: slice
    activation_function: slice
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get(self, name: str, idx: int) -> int:
        slc = getattr(self, name)
        return slc.start + idx

    def start(self, name: str) -> int:
        return getattr(self, name).start

    def stop(self, name: str) -> int:
        return getattr(self, name).stop


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
        slices[name] = slice(idx, idx + size)
        idx += size
    logit_slices = LogitSlices(**slices)
    return logit_slices


class MaskContext(BaseModel):
    logits: np.ndarray
    observation: np.ndarray
    slices: LogitSlices
    action_strategy: str
    sampling_strategy: Callable[[np.ndarray], int]
    max_layers: int
    decisions: list[int] = []  # store sampled choice for each head
    model_config = ConfigDict(arbitrary_types_allowed=True)


def build_action_add_layer_sequential(ctx: MaskContext):
    ctx.logits = mask_action_type_sequential(ctx)
    ctx.decisions.append(sample_action_from_slice(ctx, "standard_actions"))

    if ctx.decisions[0] == StandardAction.NONE.value:
        raise ValueError("No action selected, cannot proceed to add layer.")

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


def sample_action_from_slice(ctx: MaskContext, slice_name: str) -> int:
    slc = getattr(ctx.slices, slice_name)
    logits = ctx.logits[slc.start : slc.stop]
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
        new_logits[ctx.slices.standard_actions] = -np.inf
        new_logits[ctx.slices.get("standard_actions", StandardAction.ADD_LAYER.value)] = 1
        return new_logits
    modify_layer_index = ctx.slices.get("standard_actions", StandardAction.MODIFY_LAYER.value)
    remove_layer_index = ctx.slices.get("standard_actions", StandardAction.REMOVE_LAYER.value)
    new_logits[modify_layer_index] = -np.inf
    new_logits[remove_layer_index] = -np.inf
    return new_logits


def mask_indexes_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()
    latest_layer_index = get_latest_layer_index(ctx.observation)
    if latest_layer_index == ctx.max_layers - 1:
        raise MaxLayersReachedException("Maximum number of layers reached.")

    next_layer_index = latest_layer_index + 1 if latest_layer_index is not None else 0

    layer_index_start = ctx.slices.start("layer_index")
    layer_index_end = ctx.slices.stop("layer_index")

    new_logits[layer_index_start:layer_index_end] = -np.inf
    new_logits[layer_index_start + next_layer_index] = 1  # only next index is valid

    return new_logits


def mask_layer_type_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()
    if ctx.decisions[0] == StandardAction.NONE.value:
        # mask all layer types except NONE
        layer_type_start = ctx.slices.start("layer_type")
        layer_type_end = ctx.slices.stop("layer_type")
        new_logits[layer_type_start:layer_type_end] = -np.inf
        new_logits[ctx.slices.get("layer_type", LayerType.NONE.value)] = 1
        return new_logits

    linear_layer_exists = any(
        ctx.observation[i] == LayerType.LINEAR.value for i in range(0, len(ctx.observation), 7)
    )
    if linear_layer_exists:
        layer_type_start = ctx.slices.start("layer_type")
        layer_type_end = ctx.slices.stop("layer_type")
        new_logits[layer_type_start:layer_type_end] = -np.inf
        linear_index = ctx.slices.get("layer_type", LayerType.LINEAR.value)
        new_logits[linear_index] = 1  # only linear is valid

    previous_layer = get_latest_layer(ctx.observation)
    if previous_layer is None or previous_layer.layer_type != LayerType.CONV:
        # if no previous layer or previous layer is not conv, cannot add pool
        pool_index = ctx.slices.get("layer_type", LayerType.POOL.value)
        new_logits[pool_index] = -np.inf

    none_layer_index = ctx.slices.get("layer_type", LayerType.NONE.value)
    new_logits[none_layer_index] = -np.inf  # NONE is not valid when adding a layer
    return new_logits


def mask_out_channels_sequential(ctx: MaskContext):
    if ctx.decisions[2] == LayerType.LINEAR.value or ctx.decisions[2] == LayerType.NONE.value:
        # mask all out_channels
        out_channels_start = ctx.slices.start("out_channels")
        out_channels_end = ctx.slices.stop("out_channels")
        new_logits = ctx.logits.copy()
        new_logits[out_channels_start:out_channels_end] = -np.inf
        new_logits[ctx.slices.get("out_channels", OutChannels.NONE.value)] = 1
        return new_logits

    new_logits = ctx.logits.copy()
    none_index = ctx.slices.get("out_channels", OutChannels.NONE.value)
    new_logits[none_index] = -np.inf  # NONE is not valid when adding
    return new_logits


def mask_kernel_size_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()

    if ctx.decisions[2] == LayerType.NONE.value or ctx.decisions[2] == LayerType.LINEAR.value:
        # mask all kernel sizes
        kernel_size_start = ctx.slices.start("kernel_size")
        kernel_size_end = ctx.slices.stop("kernel_size")
        new_logits[kernel_size_start:kernel_size_end] = -np.inf
        new_logits[ctx.slices.get("kernel_size", KernelSize.NONE.value)] = 1
        return new_logits

    latest_output_dims = get_output_dimensions(ctx.observation)
    valid_kernels = get_valid_kernel_sizes(latest_output_dims)
    invalid_kernels = [k for k in KernelSize if k not in valid_kernels]

    for kernel in invalid_kernels:
        invalid_kernel_index = ctx.slices.get("kernel_size", kernel.value)
        new_logits[invalid_kernel_index] = -np.inf

    new_logits[ctx.slices.get("kernel_size", KernelSize.NONE.value)] = -np.inf
    return new_logits


def mask_stride_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()

    if ctx.decisions[2] == LayerType.NONE.value or ctx.decisions[2] == LayerType.LINEAR.value:
        # mask all strides
        stride_start = ctx.slices.start("stride")
        stride_end = ctx.slices.stop("stride")
        new_logits[stride_start:stride_end] = -np.inf
        new_logits[ctx.slices.get("stride", Stride.NONE.value)] = 1
        return new_logits

    kernel_size_chosen = KernelSize(ctx.decisions[4])
    latest_output_dims = get_output_dimensions(ctx.observation)
    valid_strides = get_valid_strides(latest_output_dims, kernel_size_chosen)
    invalid_strides = [s for s in Stride if s not in valid_strides]

    for stride in invalid_strides:
        invalid_stride_index = ctx.slices.get("stride", stride.value)
        new_logits[invalid_stride_index] = -np.inf

    new_logits[ctx.slices.get("stride", Stride.NONE.value)] = -np.inf
    return new_logits


def mask_linear_units_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()
    if ctx.decisions[2] != LayerType.LINEAR.value:
        new_logits[ctx.slices.linear_units] = -np.inf
        new_logits[ctx.slices.get("linear_units", LinearUnits.NONE.value)] = 1
        return new_logits

    new_logits[ctx.slices.get("linear_units", LinearUnits.NONE.value)] = -np.inf
    return new_logits


def mask_pool_mode_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()
    if ctx.decisions[2] != LayerType.POOL.value:
        new_logits[ctx.slices.pool_mode] = -np.inf
        new_logits[ctx.slices.get("pool_mode", PoolMode.NONE.value)] = 1
        return new_logits

    return ctx.logits


def mask_activation_function_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()
    # Mask NONE activation - not useful for hidden layers
    new_logits[ctx.slices.get("activation_function", ActivationFunction.NONE.value)] = -np.inf
    # Mask SOFTMAX activation - should only be used in output layer, not hidden layers
    new_logits[ctx.slices.get("activation_function", ActivationFunction.SOFTMAX.value)] = -np.inf
    return new_logits
