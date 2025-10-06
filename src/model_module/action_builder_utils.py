import enum
from typing import Callable
from pydantic import BaseModel
import numpy as np

from utils.CNNBuilder import (
    ActivationFunction,
    CNNActionSpace,
    KernelSize,
    LayerType,
    LinearUnits,
    OutChannels,
    PoolMode,
    Stride,
)


class StandardAction(enum.Enum):
    REMOVE_LAYER = 0
    MODIFY_LAYER = 1
    ADD_LAYER = 2
    DO_NOTHING = 3


class ActionStrategy(enum.Enum):
    ONLY_ADD_SEQUENTIAL = "only_add_sequential"
    ADD_REMOVE_MODIFY = "add_remove_modify"


class LogitHead(BaseModel):
    name: str
    size_fn: Callable[[int], int]


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


LOGITHEADS = [
    LogitHead(name="standard_actions", size_fn=lambda _: len(StandardAction)),
    LogitHead(name="layer_type", size_fn=lambda _: len(LayerType)),
    LogitHead(name="layer_index", size_fn=lambda max_layers: max_layers - 1),
    LogitHead(name="out_channels", size_fn=lambda _: len(OutChannels)),
    LogitHead(name="kernel_size", size_fn=lambda _: len(KernelSize)),
    LogitHead(name="stride", size_fn=lambda _: len(Stride)),
    LogitHead(name="linear_units", size_fn=lambda _: len(LinearUnits)),
    LogitHead(name="pool_mode", size_fn=lambda _: len(PoolMode)),
    LogitHead(name="activation_function", size_fn=lambda _: len(ActivationFunction)),
]


def get_logit_slices(max_layers: int):
    idx = 0
    slices = {}

    for head in LOGITHEADS:
        size = head.size_fn(max_layers)
        slices[head.name] = slice(idx, idx + size)
        idx += size

    return LogitSlices(**slices)


def add_layer_complete(
    action_output: np.ndarray,
    slices: LogitSlices,
    sampling_strategy: Callable[[np.ndarray], int],
):
    result: list[int] = []

    for key in LogitSlices.model_fields.keys():
        head = getattr(slices, key)
        logits = action_output[head]
        choice = sampling_strategy(logits)
        result.append(choice)

    return result


def apply_mask(action_output: np.ndarray, observation: np.ndarray, action_strategy: ActionStrategy):
    # latest_layer = get_latest_layer(observation)
    if action_strategy == ActionStrategy.ONLY_ADD_SEQUENTIAL:
        pass  # doe something here later
    elif action_strategy == ActionStrategy.ADD_REMOVE_MODIFY:
        pass  # doe something here later


def custom_sample_example(logits):
    probs = np.exp(logits) / np.sum(np.exp(logits))
    return int(np.random.choice(len(logits), p=probs))


def get_latest_layer(observation: np.ndarray):
    """Look for the first occurrence of -1 in the observation array with form index 7, 14, 21 ..."""
    for i in range(0, len(observation), 7):
        if observation[i] == -1 and i != 0:
            return CNNActionSpace(
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
