import torch
import torch.nn as nn
import torch.onnx
import onnx


# Custom exceptions for FFN construction
class InvalidLayerConfigError(Exception):
    pass


class UnknownActivationError(Exception):
    pass


class LayerConnectionError(Exception):
    pass


# Mapping for string activations to PyTorch modules
ACTIVATIONS = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softmax": lambda: nn.Softmax(dim=1),
    None: lambda: nn.Identity,  # no activation
}


def make_ffn(layer_config):
    """
    Build a feed-forward network (FFN) from a given config list.

    layer_config: list of tuples as shown here:
        (in_features, units, activation)
    Example:
        [(10, 20, "relu"), (20, 5, "softmax")]
    """
    layers = []

    for index, layer in enumerate(layer_config):
        # Validate layer config
        if not (isinstance(layer, tuple) and len(layer) == 3):
            raise InvalidLayerConfigError(
                f"Layer config at index {index} is not a tuple of (in_features, units, activation): {layer}"
            )

        in_f, units, act = layer

        # Check connection to next layer (if any)
        if index < len(layer_config) - 1:
            next_in_f = layer_config[index + 1][0]
            if units != next_in_f:
                raise LayerConnectionError(
                    f"Layer {index} output features ({units}) do not match next layer's input features ({next_in_f})"
                )

        layers.append(nn.Linear(in_f, units))

        if act is not None:
            try:
                act_fn = ACTIVATIONS[act.lower()]()
            except Exception:
                raise UnknownActivationError(f"Unknown activation '{act}' at layer {index}")
            layers.append(act_fn)

    return nn.Sequential(*layers)


def export_ffn_to_onnx(model, input_size, filename="ffn.onnx", opset=17):
    """
    This will export a PyTorch FFN model to a ONNX format.

    input_size: int or tuple, size of the input vector (excluding batch).
    Example: 10 → input shape will be (1, 10)
    """
    if isinstance(input_size, int):
        input_size = (1, input_size)

    dummy_input = torch.randn(*input_size)

    torch.onnx.export(
        model,
        dummy_input,  # type: ignore
        filename,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=opset,
    )

    # Verify the ONNX model
    onnx_model = onnx.load(filename)
    onnx.checker.check_model(onnx_model)
    return filename
