import enum
from typing import Callable, Tuple, Type, TypeVar
from pydantic import BaseModel, ConfigDict
import numpy as np

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.utils.network_utils import (
    Decisions,
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
LAYER_TYPE_CHOICE = 1
OUT_CHANNELS_CHOICE = 2
KERNEL_SIZE_CHOICE = 3
STRIDE_CHOICE = 4
LINEAR_UNITS_CHOICE = 5
POOL_MODE_CHOICE = 6
ACTIVATION_FUNCTION_CHOICE = 7


class MaxLayersReachedException(Exception):
    """Raised when the maximum number of layers is reached."""

    pass


class ArchitectureCompleteException(Exception):
    """Raised when the architecture is complete and no further actions can be taken."""

    pass


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

Decisions:
    A Pydantic model to store the choices made for each action type during the masking process.
    Includes a method to convert decisions to a list of integers.

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
    out_channels: LogitSlice
    kernel_size: LogitSlice
    stride: LogitSlice
    linear_units: LogitSlice
    pool_mode: LogitSlice
    activation_function: LogitSlice
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def idx(self, enum_name: str, enum_value: enum.Enum) -> int:
        """Get absolute index for a given enum name and value."""
        return getattr(self, enum_name).idx(enum_value)


def get_logit_slices():
    sizes = {
        "standard_actions": len(StandardAction),
        "layer_type": len(LayerType),
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
    sampling_strategy: Callable[[np.ndarray], int]
    max_layers: int
    decisions: Decisions
    input_dimensions: Tuple[int, int, int]
    model_config = ConfigDict(arbitrary_types_allowed=True)


def transform_decisions_to_action_indices(decisions: Decisions, slices: Slices) -> np.ndarray:
    return np.array(
        [
            decisions.action_choice.value,
            decisions.layer_type_choice.value,
            decisions.out_channels_choice.value,
            decisions.kernel_size_choice.value,
            decisions.stride_choice.value,
            decisions.linear_units_choice.value,
            decisions.pool_mode_choice.value,
            decisions.activation_function_choice.value,
        ]
    )


def transform_action_indices_to_decisions(action_indices: np.ndarray, slices: Slices):
    return Decisions(
        action_choice=StandardAction(action_indices[0]),
        layer_type_choice=LayerType(action_indices[1]),
        out_channels_choice=OutChannels(action_indices[2]),
        kernel_size_choice=KernelSize(action_indices[3]),
        stride_choice=Stride(action_indices[4]),
        linear_units_choice=LinearUnits(action_indices[5]),
        pool_mode_choice=PoolMode(action_indices[6]),
        activation_function_choice=ActivationFunction(action_indices[7]),
    )


def sample_actions(ctx: MaskContext):
    ctx.observation = ctx.observation.flatten()
    ctx.logits = mask_action_type_sequential(ctx)
    ctx.decisions.action_choice = sample_action_from_slice_v2(
        ctx, StandardAction, "standard_actions"
    )
    if ctx.decisions.action_choice == StandardAction.NONE:
        masked_logits = ctx.logits.copy()
        masked_logits[:] = -np.inf
        masked_logits[ctx.slices.standard_actions[StandardAction.NONE]] = 1
        return ctx.decisions, masked_logits

    ctx.logits = mask_layer_type_sequential(ctx)
    ctx.decisions.layer_type_choice = sample_action_from_slice_v2(ctx, LayerType, "layer_type")
    ctx.logits = mask_out_channels_sequential(ctx)
    ctx.decisions.out_channels_choice = sample_action_from_slice_v2(
        ctx, OutChannels, "out_channels"
    )
    ctx.logits = mask_kernel_size_sequential(ctx)
    ctx.decisions.kernel_size_choice = sample_action_from_slice_v2(ctx, KernelSize, "kernel_size")
    ctx.logits = mask_stride_sequential(ctx)
    ctx.decisions.stride_choice = sample_action_from_slice_v2(ctx, Stride, "stride")

    ctx.logits = mask_linear_units_sequential(ctx)
    ctx.decisions.linear_units_choice = sample_action_from_slice_v2(
        ctx, LinearUnits, "linear_units"
    )
    ctx.logits = mask_pool_mode_sequential(ctx)
    ctx.decisions.pool_mode_choice = sample_action_from_slice_v2(ctx, PoolMode, "pool_mode")
    ctx.logits = mask_activation_function_sequential(ctx)
    ctx.decisions.activation_function_choice = sample_action_from_slice_v2(
        ctx, ActivationFunction, "activation_function"
    )
    return ctx.decisions, ctx.logits


""" def sample_action_from_slice(ctx: MaskContext, slice_name: str) -> int:
    logits = ctx.logits[getattr(ctx.slices, slice_name).all]
    if np.all(logits == -np.inf):
        choice = 0  # No valid actions available
    else:
        choice = int(ctx.sampling_strategy(logits))
    return choice """


E = TypeVar("E", bound=enum.Enum)


def sample_action_from_slice_v2(ctx: MaskContext, enum_class_type: Type[E], slice_name: str) -> E:
    logits = ctx.logits[getattr(ctx.slices, slice_name).all]
    valid_indices = np.where(logits > -np.inf)[0]

    enum_class: E
    if len(valid_indices) == 0:
        enum_class = enum_class_type(0)  # No valid actions, return NONE.
    else:
        enum_class = enum_class_type(int(ctx.sampling_strategy(logits)))

    return enum_class


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

    return new_logits


def mask_layer_type_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()

    if ctx.decisions.action_choice == StandardAction.NONE:
        new_logits[ctx.slices.layer_type.all] = -np.inf
        new_logits[ctx.slices.layer_type[LayerType.NONE]] = 1
        return new_logits

    linear_layer_exists = any(
        ctx.observation[i] == LayerType.LINEAR.value for i in range(0, len(ctx.observation), 7)
    )
    if linear_layer_exists:
        new_logits[ctx.slices.layer_type.all] = -np.inf
        new_logits[ctx.slices.layer_type[LayerType.LINEAR]] = 1  # only linear is valid

        return new_logits

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
        ctx.decisions.layer_type_choice == LayerType.LINEAR
        or ctx.decisions.layer_type_choice == LayerType.NONE
    ):
        new_logits[ctx.slices.out_channels.all] = -np.inf
        new_logits[ctx.slices.out_channels[OutChannels.NONE]] = 1
        return new_logits

    new_logits[ctx.slices.out_channels[OutChannels.NONE]] = -np.inf  # NONE is not valid when adding
    return new_logits


def mask_kernel_size_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()

    if (
        ctx.decisions.layer_type_choice == LayerType.NONE
        or ctx.decisions.layer_type_choice == LayerType.LINEAR
    ):
        # mask all kernel sizes
        new_logits[ctx.slices.kernel_size.all] = -np.inf
        new_logits[ctx.slices.kernel_size[KernelSize.NONE]] = 1
        return new_logits

    latest_output_dims = get_output_dimensions(ctx.observation, ctx.input_dimensions[1:])
    valid_kernels = get_valid_kernel_sizes(latest_output_dims)
    invalid_kernels = [k for k in KernelSize if k not in valid_kernels]

    for kernel in invalid_kernels:
        invalid_kernel_index = ctx.slices.kernel_size[kernel]
        new_logits[invalid_kernel_index] = -np.inf

    new_logits[ctx.slices.kernel_size[KernelSize.NONE]] = -np.inf
    return new_logits


def mask_stride_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()

    if (
        ctx.decisions.layer_type_choice == LayerType.NONE
        or ctx.decisions.layer_type_choice == LayerType.LINEAR
    ):
        new_logits[ctx.slices.stride.all] = -np.inf
        new_logits[ctx.slices.stride[Stride.NONE]] = 1
        return new_logits

    kernel_size_chosen = KernelSize(ctx.decisions.kernel_size_choice)
    latest_output_dims = get_output_dimensions(ctx.observation, ctx.input_dimensions[1:])
    valid_strides = get_valid_strides(latest_output_dims, kernel_size_chosen)
    invalid_strides = [s for s in Stride if s not in valid_strides]

    for stride in invalid_strides:
        invalid_stride_index = ctx.slices.stride[stride.value]
        new_logits[invalid_stride_index] = -np.inf

    new_logits[ctx.slices.stride[Stride.NONE]] = -np.inf

    return new_logits


def mask_linear_units_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()

    if ctx.decisions.layer_type_choice != LayerType.LINEAR:
        new_logits[ctx.slices.linear_units.all] = -np.inf
        new_logits[ctx.slices.linear_units[LinearUnits.NONE]] = 1
        return new_logits

    new_logits[ctx.slices.linear_units[LinearUnits.NONE]] = -np.inf
    return new_logits


def mask_pool_mode_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()

    if ctx.decisions.layer_type_choice != LayerType.POOL:
        new_logits[ctx.slices.pool_mode.all] = -np.inf
        new_logits[ctx.slices.pool_mode[PoolMode.NONE]] = 1
        return new_logits

    new_logits[ctx.slices.pool_mode[PoolMode.NONE]] = -np.inf
    return new_logits


def mask_activation_function_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()
    # Mask NONE activation - not useful for hidden layers
    new_logits[ctx.slices.activation_function[ActivationFunction.NONE]] = -np.inf
    # Mask SOFTMAX activation - should only be used in output layer, not hidden layers
    new_logits[ctx.slices.activation_function[ActivationFunction.SOFTMAX]] = -np.inf

    return new_logits
