import torch
import torch.nn as nn
import torch.onnx
import onnx

# Mapping for string activations to PyTorch modules
ACTIVATIONS = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
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
    for in_f, units, act in layer_config:
        layers.append(nn.Linear(in_f, units))
        if act is not None:
            act_fn = ACTIVATIONS[act.lower()]()
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
        dummy_input,
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
