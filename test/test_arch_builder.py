import sys
import os
import pytest

import copy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

from src.utils.arch_builder import arch_builder


partial_arch1 = NetworkConfig(
    layers=[
        LayerConfig(
            layer_type=LayerType.CONV,
            out_channels=OutChannels.CH_16,
            kernel_size=KernelSize.KS_3,
            activation=ActivationFunction.RELU,
        ),
        LayerConfig(layer_type=LayerType.POOL, pool_mode=PoolMode.MAX, kernel_size=KernelSize.KS_1),
    ]
)

partial_arch2 = NetworkConfig(
    layers=[
        LayerConfig(
            layer_type=LayerType.CONV,
            out_channels=OutChannels.CH_16,
            kernel_size=KernelSize.KS_3,
            activation=ActivationFunction.RELU,
        ),
    ]
)

empty_arch = NetworkConfig(layers=[])

## [action, layerIdx, layerType, outCh, kernelSize, stride, linearU,  poolMode, actFun]
empty_arch_actions = [1, 0, 0, 0, 1, -1, -1, -1, 0]  # add conv layer
partial_arch1_actions = [1, 2, 1, -1, -1, -1, 3, -1, 1]  # add linear layer
partial_arch2_actions = [1, 1, 2, -1, 0, -1, -1, 0, 3]  # add pool layer


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
    new_arch = arch_builder(actions, partial_arch)
    assert len(new_arch.layers) == len(old_arhc.layers) + 1

    new_layer = new_arch.layers[-1]

    assert new_layer.layer_type is LayerType(actions[2])

    if actions[3] == -1:
        assert new_layer.out_channels is None
    else:
        assert new_layer.out_channels is OutChannels(actions[3])

    if actions[4] == -1:
        assert new_layer.kernel_size is None
    else:
        assert new_layer.kernel_size is KernelSize(actions[4])

    if actions[5] == -1:
        assert new_layer.stride is None
    else:
        assert new_layer.stride is Stride(actions[5])

    if actions[6] == -1:
        assert new_layer.linear_units is None
    else:
        assert new_layer.linear_units is LinearUnits(actions[6])

    if actions[7] == -1:
        assert new_layer.pool_mode is None
    else:
        assert new_layer.pool_mode is PoolMode(actions[7])

    assert new_layer.activation is ActivationFunction(actions[8])
