import pytest
import torch
from torch import nn, Tensor
from src.utils.network_utils import (
    LayerConfig,
    LayerType,
    NetworkConfig,
    PoolMode,
    OutChannels,
    KernelSize,
    Stride,
    LinearUnits,
    ActivationFunction,
)
from src.utils.graph_cnn import GraphCnn


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
            skip_connection=kwargs.get("skip_connection", None),
        )
    elif layer_type == LayerType.LINEAR:
        return LayerConfig(
            layer_type=LayerType.LINEAR,
            linear_units=kwargs.get("linear_units", LinearUnits.LU_128),
            activation=kwargs.get("activation", ActivationFunction.RELU),
            skip_connection=kwargs.get("skip_connection", None),
        )
    else:
        return LayerConfig(
            layer_type=LayerType.POOL,
            pool_mode=kwargs.get("pool_mode", PoolMode.MAX),
            kernel_size=kwargs.get("kernel_size", KernelSize.KS_3),
            stride=kwargs.get("stride", Stride.S_1),
            activation=kwargs.get("activation", ActivationFunction.RELU),
            skip_connection=kwargs.get("skip_connection", None),
        )


def test_convnet_builds_correctly(input_tensor):
    """Test: Build a simple Conv -> Pool -> Linear architecture"""
    layers = [
        make_layer(LayerType.CONV),
        make_layer(LayerType.POOL),
        make_layer(LayerType.LINEAR),
    ]
    net_config = NetworkConfig(layers=layers)
    model = GraphCnn(net_config, num_classes=10, input_dimensions=input_tensor.shape[1:])

    out = model(input_tensor)
    assert isinstance(out, Tensor)
    assert out.shape == (4, 10), f"Expected (batch, num_classes), got {out.shape}"


def test_flatten_added_when_no_linear(input_tensor):
    """Test: When no linear layers exist, final flatten + linear layer is appended automatically"""
    layers = [make_layer(LayerType.CONV), make_layer(LayerType.POOL)]
    net_config = NetworkConfig(layers=layers)
    model = GraphCnn(net_config, num_classes=5, input_dimensions=(3, 32, 32))

    out = model(input_tensor)
    assert out.shape == (4, 5)
    assert any(isinstance(m, nn.Flatten) for m in model.layers), "check that Flatten is added"
    assert any(isinstance(m, nn.Linear) for m in model.layers), "check that Linear head is added"


def test_skip_connection_conv_to_conv(input_tensor):
    """Test: Skip connection between two conv layers with matching output size and one layer in between"""
    layers = [
        make_layer(
            LayerType.CONV,
            out_channels=OutChannels.CH_16,
            kernel_size=KernelSize.KS_3,
            stride=Stride.S_1,
        ),
        make_layer(
            LayerType.CONV,
            out_channels=OutChannels.CH_32,
            kernel_size=KernelSize.KS_3,
            stride=Stride.S_1,
        ),
        make_layer(
            LayerType.CONV,
            out_channels=OutChannels.CH_32,
            kernel_size=KernelSize.KS_3,
            stride=Stride.S_1,
            skip_connection=0,
        ),
    ]
    net_config = NetworkConfig(layers=layers)
    model = GraphCnn(net_config, num_classes=10, input_dimensions=(3, 32, 32))

    out = model(input_tensor)
    assert out.shape == (4, 10)
    assert len(model.projections) == 1, "Should have one skip projection"


def test_skip_connection_conv_to_linear():
    """Test: Conv -> Linear skip connection handled via projection"""
    input_tensor = torch.randn(4, 3, 16, 16)

    layers = [
        make_layer(
            LayerType.CONV,
            out_channels=OutChannels.CH_16,
            kernel_size=KernelSize.KS_3,
            stride=Stride.S_2,
        ),
        make_layer(
            LayerType.LINEAR,
            linear_units=LinearUnits.LU_128,
        ),
        make_layer(
            LayerType.LINEAR,
            linear_units=LinearUnits.LU_128,
            skip_connection=0,
        ),
    ]

    net_config = NetworkConfig(layers=layers)
    model = GraphCnn(net_config, num_classes=10, input_dimensions=(3, 16, 16))

    out = model(input_tensor)

    # Assertions
    assert out.shape == (4, 10)
    assert len(model.projections) == 1, "Expected one projection for Conv -> Linear skip"
    proj = next(iter(model.projections.values()))
    assert isinstance(proj, (nn.Flatten, nn.Sequential))


def test_infer_out_channels_linear_and_conv():
    """Ensure _infer_out_channels works for convs and linears"""
    model = GraphCnn(NetworkConfig(layers=[]), num_classes=10, input_dimensions=(3, 32, 32))

    conv = nn.Conv2d(3, 8, 3)
    linear = nn.Linear(16, 4)
    seq = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU())

    assert model._infer_out_channels(conv) == 8
    assert model._infer_out_channels(linear) == 4
    assert model._infer_out_channels(seq) == 8


def test_forward_pass_runs_without_error(input_tensor):
    layers = [make_layer(LayerType.CONV), make_layer(LayerType.POOL), make_layer(LayerType.LINEAR)]
    net_config = NetworkConfig(layers=layers)
    model = GraphCnn(net_config, num_classes=10, input_dimensions=(3, 32, 32))
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
    model = GraphCnn(net_config, num_classes=10, input_dimensions=(3, 32, 32))

    # Check that no BatchNorm layers are present
    assert not any(isinstance(m, nn.BatchNorm2d) for m in model.layers), (
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
    model = GraphCnn(net_config, num_classes=10, input_dimensions=(3, 32, 32))

    print(model.layers)

    # Check that BatchNorm layers are present after Conv layers
    for seq in model.layers:
        if isinstance(seq, nn.Sequential) and any(isinstance(m, nn.Conv2d) for m in seq):
            assert any(isinstance(m, nn.BatchNorm2d) for m in seq), (
                "BatchNorm should be added after Conv layers"
            )


def test_batch_norm_with_skip_connections(input_tensor):
    """Test: BatchNorm is added correctly when skip connections are present"""
    layers = [
        make_layer(LayerType.CONV),
        make_layer(LayerType.CONV),
        make_layer(LayerType.CONV, skip_connection=0),
        make_layer(LayerType.CONV, skip_connection=2),
        make_layer(LayerType.LINEAR),
    ]
    net_config = NetworkConfig(layers=layers)
    model = GraphCnn(net_config, num_classes=10, input_dimensions=(3, 32, 32))

    # Check that BatchNorm layers are present after Conv layers
    for seq in model.layers:
        if isinstance(seq, nn.Sequential) and any(isinstance(m, nn.Conv2d) for m in seq):
            assert any(isinstance(m, nn.BatchNorm2d) for m in seq), (
                "BatchNorm should be added after Conv layers"
            )
