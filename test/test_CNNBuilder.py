import sys
import os
import pytest
import torch.nn as nn
import torch


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.CNNBuilder import CNNBuilder, InvalidLayerConfigError, CNNExportError


# ---------------------------- Testing the Initialization


@pytest.mark.parametrize(
    "layer_config, in_channels, input_size, num_classes",
    [
        ([], 1, (28, 28), 26),  # empty config, default grayscale
        (
            [{"layer_type": "conv", "out_channels": 32, "kernel_size": 3}],
            3,
            (32, 32),
            10,
        ),  # RGB example
        ([{"layer_type": "linear", "linear_units": 128}], 1, (28, 28), 5),  # only linear
    ],
)
def test_cnn_builder_initialization(layer_config, in_channels, input_size, num_classes):
    cnn = CNNBuilder(
        layer_config, in_channels=in_channels, input_size=input_size, num_classes=num_classes
    )

    # Check attributes
    assert cnn.layer_config == layer_config
    assert cnn.in_channels == in_channels
    assert cnn.input_size == input_size
    assert cnn.num_classes == num_classes

    # Initially model should be None
    assert cnn.model is None


# ---------------------------- Testing Layer Config Handling


def test_valid_layer_config():
    layer_config = [
        {"layer_type": "conv", "out_channels": 16, "kernel_size": 3, "activation": "relu"},
        {"layer_type": "pool", "pool_mode": "max", "kernel_size": 2},
        {"layer_type": "linear", "linear_units": 128, "activation": "tanh"},
    ]

    builder = CNNBuilder(layer_config, in_channels=1, input_size=(28, 28), num_classes=10)
    model = builder.build()

    # Check that the model is nn.Sequential
    assert isinstance(model, nn.Sequential)

    # Only check types without assuming positions
    types = [type(layer) for layer in model]
    assert nn.Conv2d in types
    assert nn.ReLU in types
    assert nn.MaxPool2d in types
    assert nn.Flatten in types
    assert nn.Linear in types
    assert nn.Tanh in types


@pytest.mark.parametrize(
    "layer_config,raises",
    [
        ([{"layer_type": "conv", "out_channels": 16, "kernel_size": 3}], False),
        ([{"layer_type": "pool", "pool_mode": "max", "kernel_size": 2}], False),
        ([{"layer_type": "linear", "linear_units": 128}], False),
        ([{"layer_type": "unknown"}], True),
        ([{"layer_type": "pool", "pool_mode": "invalid", "kernel_size": 2}], True),
    ],
)
def test_layer_config_handling(layer_config, raises):
    builder = CNNBuilder(layer_config, in_channels=1, input_size=(28, 28), num_classes=10)
    if raises:
        with pytest.raises(InvalidLayerConfigError):
            builder.build()
    else:
        model = builder.build()
        assert isinstance(model, nn.Sequential)


def test_multiple_linear_layers():
    config = [
        {"layer_type": "linear", "linear_units": 16, "activation": "relu"},
        {"layer_type": "linear", "linear_units": 32, "activation": "tanh"},
    ]
    builder = CNNBuilder(config, input_size=(2, 2))
    model = builder.build()
    # Check flatten occurs only once
    assert any(isinstance(layer, nn.Flatten) for layer in model)
    # Check both linear layers exist
    linear_outs = [layer.out_features for layer in model if isinstance(layer, nn.Linear)]
    assert linear_outs == [16, 32, builder.num_classes]  # final classifier included


# -------------------------- test Convolution Layers
def test_conv_layer_properties():
    config = [{"layer_type": "conv", "out_channels": 8, "kernel_size": 3, "activation": "relu"}]
    builder = CNNBuilder(config, in_channels=1, input_size=(28, 28), num_classes=10)
    model = builder.build()

    conv = next(layer for layer in model if isinstance(layer, nn.Conv2d))
    assert conv.in_channels == 1
    assert conv.out_channels == 8
    assert conv.kernel_size == (3, 3)
    assert isinstance(model[1], nn.ReLU)


# ---------------------- test Pooling Layers
def test_pooling_layers():
    config = [{"layer_type": "pool", "pool_mode": "avg", "kernel_size": 2}]
    builder = CNNBuilder(config)
    model = builder.build()
    pool = next(layer for layer in model if isinstance(layer, nn.AvgPool2d))
    assert pool.kernel_size == 2
    assert pool.stride == 2  # default stride = kernel


# ----------------- test Linear Layers
def test_linear_layers():
    config = [{"layer_type": "linear", "linear_units": 32, "activation": "tanh"}]
    builder = CNNBuilder(config, in_channels=1, input_size=(4, 4), num_classes=10)
    model = builder.build()

    # Find the first linear layer
    linear_idx = next(i for i, layer in enumerate(model) if isinstance(layer, nn.Linear))
    linear = model[linear_idx]
    assert linear.out_features == 32

    # Check that the activation immediately follows the linear layer
    activation = model[linear_idx + 1]
    assert isinstance(activation, nn.Tanh)


# --------- test Final Classifier
def test_final_classifier():
    config = []  # no linear layers
    builder = CNNBuilder(config, in_channels=1, input_size=(28, 28), num_classes=26)
    model = builder.build()
    assert isinstance(model[-1], nn.Linear)
    assert model[-1].out_features == 26


# ----------- test Model Build & Forward
def test_forward_pass():
    config = [
        {"layer_type": "conv", "out_channels": 4, "kernel_size": 3},
        {"layer_type": "linear", "linear_units": 8},
    ]
    builder = CNNBuilder(config, in_channels=1, input_size=(8, 8), num_classes=5)
    model = builder.build()

    x = torch.randn(2, 1, 8, 8)
    out = model(x)
    assert out.shape == (2, 5)


# --------------- test act functions
@pytest.mark.parametrize(
    "act_type,act_class",
    [("relu", nn.ReLU), ("tanh", nn.Tanh), ("softmax", nn.Softmax), ("none", nn.Identity)],
)
def test_activation_mapping(act_type, act_class):
    config = [{"layer_type": "linear", "linear_units": 4, "activation": act_type}]
    builder = CNNBuilder(config, in_channels=1, input_size=(4, 4), num_classes=2)
    model = builder.build()

    # Scan all layers for a matching activation instance
    assert any(isinstance(layer, act_class) for layer in model), f"{act_type} not found in model"


# ---------------------- test edge case
def test_edge_cases():
    # Empty config
    builder = CNNBuilder([])
    model = builder.build()
    assert isinstance(model[-1], nn.Linear)

    # Only conv/pool
    config = [
        {"layer_type": "conv", "out_channels": 4, "kernel_size": 3},
        {"layer_type": "pool", "pool_mode": "max", "kernel_size": 2},
    ]
    builder = CNNBuilder(config, input_size=(8, 8))
    model = builder.build()
    assert isinstance(model[-1], nn.Linear)


# ------------------------- # overall test
@pytest.mark.parametrize("input_size,num_classes", [((28, 28), 10), ((32, 32), 26), ((16, 16), 5)])
def test_integration(input_size, num_classes):
    config = [
        {"layer_type": "conv", "out_channels": 4, "kernel_size": 3},
        {"layer_type": "linear", "linear_units": 8},
    ]
    builder = CNNBuilder(config, in_channels=1, input_size=input_size, num_classes=num_classes)
    model = builder.build()
    import torch

    x = torch.randn(1, 1, *input_size)
    out = model(x)
    assert out.shape == (1, num_classes)


# ------------------------------------------------------- Testing export_to_onnx
def test_export_after_build(tmp_path):
    config = [
        {"layer_type": "conv", "out_channels": 4, "kernel_size": 3},
        {"layer_type": "linear", "linear_units": 8},
    ]
    builder = CNNBuilder(config, in_channels=1, input_size=(8, 8), num_classes=2)
    builder.build()

    filename = tmp_path / "test_model.onnx"
    exported_path = builder.export_to_onnx(input_size=(1, 8, 8), filename=str(filename))

    assert os.path.exists(exported_path)
    assert exported_path.endswith(".onnx")


def test_export_without_build_raises():
    builder = CNNBuilder([])
    with pytest.raises(CNNExportError, match="Build the model first"):
        builder.export_to_onnx()


def test_export_auto_filename(tmp_path):
    builder = CNNBuilder([{"layer_type": "linear", "linear_units": 4}], input_size=(2, 2))
    builder.build()

    # Export to ONNX, get the full path
    path = builder.export_to_onnx(input_size=(1, 2, 2))

    try:
        # Check that the file exists
        assert os.path.exists(path)

        # Check that the filename ends with .onnx
        assert path.endswith(".onnx")

        # Optionally: check that the path contains the saved_models folder
        assert "saved_models" in str(path)
    finally:
        # Clean up: delete the file after test
        if os.path.exists(path):
            os.remove(path)
