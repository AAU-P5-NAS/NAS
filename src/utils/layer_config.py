import enum
from typing import Optional, TYPE_CHECKING
import numpy as np
from pydantic import BaseModel, model_validator
import torch.nn as nn

if TYPE_CHECKING:
    # Only imported for type checking to avoid circular imports at runtime.
    from src.agent.action_masking.action_masking_utils import Decisions

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
    # LU_256 = 3  # 256
    # LU_512 = 4  # 512

    def to_units(self):
        mapping = [None, 64, 128]
        return mapping[self.value]


class OutChannels(enum.IntEnum):
    NONE = 0
    CH_16 = 1  # 16
    CH_32 = 2  # 32
    CH_64 = 3  # 64
    # CH_128 = 4  # 128
    # CH_256 = 5  # 256

    def to_channels(self):
        mapping = [None, 16, 32, 64]
        return mapping[self.value]


class KernelSize(enum.IntEnum):
    NONE = 0
    # KS_1 = 1  # 1
    KS_3 = 1  # 3
    # KS_5 = 3  # 5

    def to_kernel(self):
        mapping = [None, 3]
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
    # AVG = 2  # "avg"

    def to_pmode(self):
        mapping = [None, "max"]
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
    def from_decisions(cls, actions:"Decisions"):
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