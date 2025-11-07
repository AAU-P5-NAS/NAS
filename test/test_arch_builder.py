import sys
import os
import pytest

import copy

from src.utils.network_utils import (
    LayerType,
    OutChannels,
    KernelSize,
    LinearUnits,
    ActivationFunction,
    PoolMode,
    NetworkConfig,
    LayerConfig,
    Stride,
)

from src.utils.cnn_builder import CNNBuilder

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

NUM_CLASSES = 28
dimensions = (3, 28, 28)

partial_arch1 = NetworkConfig(
    layers=[
        LayerConfig(
            layer_type=LayerType.LINEAR,
            activation=ActivationFunction.RELU,
            linear_units=LinearUnits.LU_256,
        )
    ]
)

partial_arch2 = NetworkConfig(
    layers=[
        LayerConfig(
            layer_type=LayerType.POOL,
            kernel_size=KernelSize.KS_3,
            activation=ActivationFunction.RELU,
            pool_mode=PoolMode.AVG,
        ),
    ]
)

empty_arch = NetworkConfig(layers=[])

## [action, layerIdx, layerType, outCh, kernelSize, stride, linearU,  poolMode, actFun]
empty_arch_actions = [2, 0, 1, 0, 1, -1, -1, -1, 0]  # add conv layer
partial_arch1_actions = [2, 2, 2, -1, -1, -1, 3, -1, 1]  # add linear layer
partial_arch2_actions = [2, 1, 3, -1, 2, -1, -1, 2, 1]  # add pool layer


@pytest.mark.parametrize(
    "actions, partial_arch",
    [
        (empty_arch_actions, empty_arch),
        (partial_arch1_actions, partial_arch1),
        (partial_arch2_actions, partial_arch2),
    ],
)
def test_arch_builder_given_valid_input(actions: list[int], partial_arch: NetworkConfig):
    old_arhc = copy.deepcopy(partial_arch)
    new_arch = CNNBuilder(partial_arch, NUM_CLASSES, dimensions)
    assert len(new_arch.rl_config.layers) == len(old_arhc.layers)

    # If no layers, then we can't check the layer
    if not new_arch.rl_config.layers:
        return

    new_layer = new_arch.rl_config.layers[-1]

    assert new_layer.layer_type is LayerType(actions[2])

    if actions[3] == -1:
        assert new_layer.out_channels is OutChannels.NONE
    else:
        assert new_layer.out_channels is OutChannels(actions[3])

    if actions[4] == -1:
        assert new_layer.kernel_size is KernelSize.NONE
    else:
        assert new_layer.kernel_size is KernelSize(actions[4])

    if actions[5] == -1:
        assert new_layer.stride is Stride.NONE
    else:
        assert new_layer.stride is Stride(actions[5])

    if actions[6] == -1:
        assert new_layer.linear_units is LinearUnits.NONE
    else:
        assert new_layer.linear_units is LinearUnits(actions[6])

    if actions[7] == -1:
        assert new_layer.pool_mode is PoolMode.NONE
    else:
        assert new_layer.pool_mode is PoolMode(actions[7])

    assert new_layer.activation is ActivationFunction(actions[8])
