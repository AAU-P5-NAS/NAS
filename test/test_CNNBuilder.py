import sys
import os
import pytest
import torch.nn as nn
import onnx


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
    InvalidLayerConfigError,
    InvalidLayerOrderError,
    CNNExportError,
)

from src.utils.cnn_builder import CNNBuilder, flatten_cnn_config


@pytest.fixture
def valid_rl_config():
    """Return a standard valid RLConfig."""
    return NetworkConfig(
        layers=[
            LayerConfig(
                layer_type=LayerType.CONV,
                out_channels=OutChannels.CH_16,
                kernel_size=KernelSize.KS_3,
                activation=ActivationFunction.RELU,
            ),
            LayerConfig(
                layer_type=LayerType.POOL,
                pool_mode=PoolMode.MAX,
                kernel_size=KernelSize.KS_1,
            ),
            LayerConfig(
                layer_type=LayerType.LINEAR,
                linear_units=LinearUnits.LU_64,
                activation=ActivationFunction.TANH,
            ),
        ]
    )


def test_valid_cnn_build(valid_rl_config):
    """Test a valid RLConfig builds correctly"""
    cnn_builder = CNNBuilder(valid_rl_config, input_size=(28, 28), num_classes=26)
    model = cnn_builder.build()

    assert isinstance(model, nn.Sequential)
    assert isinstance(model[-1], nn.Linear)
    assert model[-1].out_features == 26


def test_invalid_layer_config_raises_error():
    """Test CNNActionSpace raises InvalidLayerConfigError if conv layer missing kernel"""
    with pytest.raises(InvalidLayerConfigError):
        LayerConfig(
            layer_type=LayerType.CONV,
            out_channels=OutChannels.CH_16,  # kernel_size is missing
        )


def test_invalid_layer_order_raises_error():
    """Test RLConfig raises InvalidLayerOrderError if conv appears after linear"""
    with pytest.raises(InvalidLayerOrderError):
        NetworkConfig(
            layers=[
                LayerConfig(
                    layer_type=LayerType.LINEAR,
                    linear_units=LinearUnits.LU_64,
                ),
                LayerConfig(
                    layer_type=LayerType.CONV,
                    out_channels=OutChannels.CH_16,
                    kernel_size=KernelSize.KS_3,
                ),
            ]
        )


@pytest.mark.parametrize("save_separate", [True, False])
def test_onnx_export(valid_rl_config, tmp_path, save_separate):
    """Test ONNX export works and creates a valid file"""
    builder = CNNBuilder(valid_rl_config)
    builder.build()

    path = builder.export_to_onnx(save_in_seperate_file=save_separate)

    assert os.path.exists(path)

    model = onnx.load(path)
    onnx.checker.check_model(model)

    os.remove(path)


def test_onnx_export_raises_CNNExportError(valid_rl_config):
    builder = CNNBuilder(valid_rl_config)

    with pytest.raises(CNNExportError):
        builder.export_to_onnx()


@pytest.mark.parametrize("size", [3, 6])
def test_cnn_config_to_flatt(valid_rl_config, size):
    flatten = flatten_cnn_config(valid_rl_config, size)

    assert len(flatten) == size * 7, f"Expected {size * 7}, got {len(flatten)}"
    assert all(isinstance(x, int) for x in flatten)
    assert all(x >= -1 for x in flatten)


@pytest.mark.parametrize("size", [2])
def test_cnn_config_to_flatt_fails(valid_rl_config, size):
    with pytest.raises(ValueError):
        flatten_cnn_config(valid_rl_config, size)
