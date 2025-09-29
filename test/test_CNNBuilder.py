import sys
import os
import pytest
import torch.nn as nn
import onnx


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.CNNBuilder import (
    CNNBuilder,
    InvalidLayerConfigError,
    CNNExportError,
    InvalidLayerOrderError,
    RLConfig,
    CNNActionSpace,
    LayerType,
    OutChannels,
    KernelSize,
    LinearUnits,
    ActivationFunction,
    PoolMode,
)


@pytest.fixture
def valid_rl_config():
    """Return a standard valid RLConfig."""
    return RLConfig(
        layers=[
            CNNActionSpace(
                layer_type=LayerType.CONV,
                out_channels=OutChannels.CH_16,
                kernel_size=KernelSize.KS_3,
                activation=ActivationFunction.RELU,
            ),
            CNNActionSpace(
                layer_type=LayerType.POOL,
                pool_mode=PoolMode.MAX,
                kernel_size=KernelSize.KS_1,
            ),
            CNNActionSpace(
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
        CNNActionSpace(
            layer_type=LayerType.CONV,
            out_channels=OutChannels.CH_16,  # kernel_size is missing
        )


def test_invalid_layer_order_raises_error():
    """Test RLConfig raises InvalidLayerOrderError if conv appears after linear"""
    with pytest.raises(InvalidLayerOrderError):
        RLConfig(
            layers=[
                CNNActionSpace(
                    layer_type=LayerType.LINEAR,
                    linear_units=LinearUnits.LU_64,
                ),
                CNNActionSpace(
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
