import sys
import os
import pytest
import torch.nn as nn


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.ffn_utils import (
    make_ffn,
    InvalidLayerConfigError,
    UnknownActivationError,
    LayerConnectionError,
)


# Test layer types in FFNs
@pytest.mark.parametrize(
    "layer_config, expected_types",
    [
        ([(10, 20, "relu")], [nn.Linear, nn.ReLU]),
        ([(10, 20, "sigmoid")], [nn.Linear, nn.Sigmoid]),
        ([(10, 20, "tanh")], [nn.Linear, nn.Tanh]),
        ([(10, 20, "softmax")], [nn.Linear, nn.Softmax]),
        ([(10, 20, None)], [nn.Linear]),  # No activation
        ([(10, 20, "relu"), (20, 5, "softmax")], [nn.Linear, nn.ReLU, nn.Linear, nn.Softmax]),
    ],
)
def test_layer_types(layer_config, expected_types):
    model = make_ffn(layer_config)
    layers = list(model)
    assert len(layers) == len(expected_types)
    for layer, expected_type in zip(layers, expected_types):
        assert isinstance(layer, expected_type)


# Layer connection tests (number of out_features of one layer should match in_features of the next)
@pytest.mark.parametrize(
    "layer_config",
    [
        [(10, 20, "relu"), (20, 30, "sigmoid"), (30, 5, "softmax")],
        [(5, 5, "tanh"), (5, 5, "relu")],
        [(8, 16, "relu"), (16, 8, None), (8, 4, "sigmoid")],
    ],
)
def test_layer_connections(layer_config):
    model = make_ffn(layer_config)
    layers = [layer for layer in model if isinstance(layer, nn.Linear)]
    for i in range(len(layers) - 1):
        assert layers[i].out_features == layers[i + 1].in_features


# Invalid acrchitecture tests
@pytest.mark.parametrize(
    "invalid_config",
    [
        [(10, 20)],  # Missing activation
        [(10, 20, "relu"), (15, 5, "softmax")],  # Mismatched in/out features
        [(10, 20, "unknown_activation")],  # Unknown activation
        "not_a_list",  # Not a list
        [(10, 20, "relu"), (20,)],  # Second layer invalid
    ],
)
def test_invalid_architectures(invalid_config):
    with pytest.raises((InvalidLayerConfigError, UnknownActivationError, LayerConnectionError)):
        make_ffn(invalid_config)
