import sys
import os
import pytest
import torch
import torch.nn as nn

# Ensure the project root is in sys.path so 'src' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.ffn_utils import make_ffn


# ------------- test make_ffn
def test_make_ffn_structure():
    config = [(10, 20, "relu"), (20, 5, "softmax")]
    model = make_ffn(config)

    # Check layer sequence: Linear -> ReLU -> Linear -> Softmax
    assert isinstance(model[0], nn.Linear)
    assert model[0].in_features == 10
    assert model[0].out_features == 20

    assert isinstance(model[1], nn.ReLU)

    assert isinstance(model[2], nn.Linear)
    assert model[2].in_features == 20
    assert model[2].out_features == 5

    assert isinstance(model[3], nn.Softmax)
    assert model[3].dim == 1


def test_make_ffn_forward_pass():
    config = [(4, 3, "tanh"), (3, 2, None)]
    model = make_ffn(config)

    x = torch.randn(5, 4)  # batch of 5
    out = model(x)
    assert out.shape == (5, 2)


def test_invalid_activation():
    config = [(3, 2, "not_an_activation")]
    with pytest.raises(KeyError):
        make_ffn(config)


# ------ test export_ffn_to_onnx -----------
