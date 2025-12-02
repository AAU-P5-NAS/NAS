import torch
from torch import Tensor, nn
import pytest

from src.utils.network_config import (
    NetworkConfig,
)
from src.utils.architecture import Architecture
from src.utils.layer_config import (
    LayerConfig,
    LayerType,
    OutChannels,
    KernelSize,
    Stride,
    LinearUnits,
    PoolMode,
    ActivationFunction,
)


@pytest.fixture
def input_tensor():
    """(batch_size, channels, height, width)"""
    return torch.randn(4, 3, 32, 32)


def make_layer(layer_type, **kwargs) -> LayerConfig:
    """Helper to construct a single LayerConfig"""
    if layer_type == LayerType.CONV:
        return LayerConfig(
            layer_type=LayerType.CONV,
            out_channels=kwargs.get("out_channels", OutChannels.CH_16),
            kernel_size=kwargs.get("kernel_size", KernelSize.KS_3),
            stride=kwargs.get("stride", Stride.S_1),
            activation=kwargs.get("activation", ActivationFunction.RELU),
        )
    elif layer_type == LayerType.LINEAR:
        return LayerConfig(
            layer_type=LayerType.LINEAR,
            linear_units=kwargs.get("linear_units", LinearUnits.LU_128),
            activation=kwargs.get("activation", ActivationFunction.RELU),
        )
    else:
        return LayerConfig(
            layer_type=LayerType.POOL,
            pool_mode=kwargs.get("pool_mode", PoolMode.MAX),
            kernel_size=kwargs.get("kernel_size", KernelSize.KS_3),
            stride=kwargs.get("stride", Stride.S_1),
            activation=kwargs.get("activation", ActivationFunction.RELU),
        )


def test_convnet_builds_correctly(input_tensor):
    """Test: Build a simple Conv -> Pool -> Linear architecture"""
    layers = [
        make_layer(LayerType.CONV),
        make_layer(LayerType.POOL),
        make_layer(LayerType.LINEAR),
    ]
    net_config = NetworkConfig(layers=layers)
    model = Architecture(net_config, num_classes=10, input_dimensions=input_tensor.shape[1:])

    out = model(input_tensor)
    assert isinstance(out, Tensor)
    assert out.shape == (4, 10), f"Expected (batch, num_classes), got {out.shape}"


def test_flatten_added_when_no_linear(input_tensor):
    """Test: When no linear layers exist, final flatten + linear layer is appended automatically"""
    layers = [make_layer(LayerType.CONV), make_layer(LayerType.POOL)]
    net_config = NetworkConfig(layers=layers)
    model = Architecture(net_config, num_classes=5, input_dimensions=(3, 32, 32))

    out = model(input_tensor)
    assert out.shape == (4, 5)
    assert any(isinstance(m, nn.Flatten) for m in model.model), "check that Flatten is added"
    assert any(isinstance(m, nn.Linear) for m in model.model), "check that Linear head is added"

def test_forward_pass_runs_without_error(input_tensor):
    layers = [make_layer(LayerType.CONV), make_layer(LayerType.POOL), make_layer(LayerType.LINEAR)]
    net_config = NetworkConfig(layers=layers)
    model = Architecture(net_config, num_classes=10, input_dimensions=(3, 32, 32))
    with torch.no_grad():
        out = model(input_tensor)
    assert out.shape == (4, 10)

def test_batch_norm_not_added_for_small_nets(input_tensor):
    """Test: BatchNorm is not added for small networks (less than 4 conv layers)"""
    layers = [
        make_layer(LayerType.CONV),
        make_layer(LayerType.POOL),
        make_layer(LayerType.CONV),
        make_layer(LayerType.POOL),
        make_layer(LayerType.CONV),
        make_layer(LayerType.POOL),
        make_layer(LayerType.LINEAR),
    ]
    net_config = NetworkConfig(layers=layers)
    model = Architecture(net_config, num_classes=10, input_dimensions=(3, 32, 32))

    # Check that no BatchNorm layers are present
    assert not any(isinstance(m, nn.BatchNorm2d) for m in model.model), (
        "No BatchNorm should be added"
    )

def test_batch_norm_added_for_large_nets(input_tensor):
    """Test: BatchNorm is added for networks with 4 or more conv layers"""
    layers = [
        make_layer(LayerType.CONV),
        make_layer(LayerType.CONV),
        make_layer(LayerType.CONV),
        make_layer(LayerType.CONV),
        make_layer(LayerType.LINEAR),
    ]
    net_config = NetworkConfig(layers=layers)
    model = Architecture(net_config, num_classes=10, input_dimensions=(3, 32, 32))

    # Check that BatchNorm layers are present after Conv layers
    assert any(isinstance(m, nn.BatchNorm2d) for m in model.model)

def count_dropouts_and_find_flatten(model: nn.Sequential):
    """Helper: count nn.Dropout occurrences and find first Flatten index and whether Dropout follows."""
    total_dropouts = 0
    flatten_followed_by_dropout = False
   
    layers = list(model)

    for i, mod in enumerate(layers):

        # Count Dropout
        if isinstance(mod, nn.Dropout):
            total_dropouts += 1

        # Detect Flatten → Dropout
        if isinstance(mod, nn.Flatten):
            if i + 1 < len(layers) and isinstance(layers[i + 1], nn.Dropout):
                flatten_followed_by_dropout = True

    return total_dropouts, flatten_followed_by_dropout

def test_dropout_after_flatten_when_linear_layer():
    # Build a NetworkConfig with a single LINEAR layer (should cause flatten+dropout inside sequential)
    net = NetworkConfig(layers=[make_layer(LayerType.LINEAR)])

    model = Architecture(net, num_classes=LinearUnits.LU_64.to_units(), input_dimensions=(1, 28, 28))

    total_dropouts, flatten_followed = count_dropouts_and_find_flatten(model.model)

    assert total_dropouts == 1, f"Expected exactly one Dropout, found {total_dropouts}"
    assert flatten_followed, "Expected Dropout to appear immediately after Flatten"


def test_dropout_appended_when_no_linear_layers():
    # NetworkConfig with a CONV then POOL (no linear) -> flatten+Dropout should be appended at the end
    net = NetworkConfig(layers=[make_layer(LayerType.CONV), make_layer(LayerType.POOL)])

    model = Architecture(net, num_classes=10, input_dimensions=(1, 28, 28))

    total_dropouts, flatten_followed = count_dropouts_and_find_flatten(model.model)

    assert total_dropouts == 1, f"Expected exactly one Dropout, found {total_dropouts}"
    assert flatten_followed, "Expected Dropout to appear immediately after Flatten"
