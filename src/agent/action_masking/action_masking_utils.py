import enum
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Type, TypeVar
from pydantic import BaseModel, ConfigDict
import numpy as np

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if TYPE_CHECKING:
    # Only imported for type checking to avoid circular imports at runtime.
    from src.utils.layer_config import StandardAction, LayerType, OutChannels, KernelSize, Stride, LinearUnits, PoolMode, ActivationFunction



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
    skip_connection: LogitSlice
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def idx(self, enum_name: str, enum_value: enum.Enum) -> int:
        """Get absolute index for a given enum name and value."""
        return getattr(self, enum_name).idx(enum_value)


def get_logit_slices(max_layers: int):
    sizes = {
        "standard_actions": len(StandardAction),
        "layer_type": len(LayerType),
        "out_channels": len(OutChannels),
        "kernel_size": len(KernelSize),
        "stride": len(Stride),
        "linear_units": len(LinearUnits),
        "pool_mode": len(PoolMode),
        "activation_function": len(ActivationFunction),
        "skip_connection": max_layers,
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
    action_count: int
    model_config = ConfigDict(arbitrary_types_allowed=True)


NO_ACTION_DECISIONS = Decisions(
    action_choice=StandardAction.NONE,
    layer_type_choice=LayerType.NONE,
    out_channels_choice=OutChannels.NONE,
    kernel_size_choice=KernelSize.NONE,
    stride_choice=Stride.NONE,
    linear_units_choice=LinearUnits.NONE,
    pool_mode_choice=PoolMode.NONE,
    activation_function_choice=ActivationFunction.NONE,
    skip_connection_choice=None,
)


E = TypeVar("E", bound=enum.Enum)


def transform_decisions_to_action_indices(decisions: Decisions, slices: Slices, max_layers:int) -> np.ndarray:
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
            decisions.skip_connection_choice
            if decisions.skip_connection_choice is not None
            else max_layers-1,
        ]
    )


def sample_action_for_slice(ctx: MaskContext, enum_class_type: Type[E], slice_name: str) -> E:
    logits = ctx.logits[getattr(ctx.slices, slice_name).all]
    valid_indices = np.where(logits > -np.inf)[0]
    enum_class: E
    if len(valid_indices) == 0:
        enum_class = enum_class_type(0)  # No valid actions, return NONE.
    else:
        enum_class = enum_class_type(int(ctx.sampling_strategy(logits)))

    return enum_class


def sample_skip_connection(ctx: MaskContext):
    logits = ctx.logits[ctx.slices.skip_connection.all]
    valid_indices = np.where(logits > -np.inf)[0]

    length = len(valid_indices)

    if length <= 1:
        return None

    sampled_index = ctx.sampling_strategy(logits)
    if sampled_index == ctx.max_layers - 1:
        return None  # no skip connection

    return sampled_index


def standard_stochastic_sampling(logits: np.ndarray) -> int:
    """Samples an index from the logits using a softmax distribution."""
    exp_logits = np.exp(logits - np.max(logits))  # ensure positive numbers
    probs = exp_logits / exp_logits.sum()  # compute probabilities
    return np.random.choice(len(logits), p=probs)  # sample index based on probs and return idx


def max_sampling(logits: np.ndarray) -> int:
    """Selects the index with the highest logit value."""
    return int(np.argmax(logits))




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

