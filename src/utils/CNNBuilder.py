import torch.nn as nn
import torch.onnx
import onnx
import os

"""
NOTE: 
in standard CNN design, conv and linear layers are not usually mixed. The typical structure is:
Input → Conv/Pool layers → Flatten → Linear (fully connected) layers → Output


If the RL-generated config were something like:
[
    {"layer_type": "conv", "out_channels": 32, "kernel_size": 3},
    {"layer_type": "linear", "linear_units": 128},
    {"layer_type": "conv", "out_channels": 64, "kernel_size": 3}
]

then the below code would run into problems:

After the first linear layer, nn.Flatten() has already been applied.
- The next conv layer expects a 3D input [batch, channels, height, width], but it now receives a 1D vector [batch, features].
- This mismatch will cause a runtime error when building or running the model.

Hence, the RL config must have the structure:
[conv/pool layers ...] → [linear layers ...] → final classifier

"""


# the RL sohuld use this action space to make a layer_config
# CNN_ACTION_SPACE = {
#     "layer_type": ["conv", "pool", "linear"],
#     "out_channels": [16, 32, 64, 128],
#     "kernel_size": [1, 3, 5],
#     "stride": [1, 2],
#     "pool_mode": ["max", "avg"],
#     "activation": ["relu", "tanh", softmax, "none"],
#     "linear_units": [64, 128, 256, 512],
# }

# the cofig from the RL should look like this

# Example layer_config (from CNN_ACTION_SPACE):
#     [
#         {"layer_type": "conv", "out_channels": 32, "kernel_size": 3, "stride": 1, "activation": "relu"},
#         {"layer_type": "pool", "pool_mode": "max", "kernel_size": 2},
#         {"layer_type": "linear", "linear_units": 128, "activation": "tanh"},
#     ]


# Activation mapping
ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "softmax": lambda: nn.Softmax(dim=1),  # lambda to pass dim argument
    "none": lambda: nn.Identity(),
}


class InvalidLayerConfigError(Exception):
    """Raised for any error in build() method"""

    pass


class CNNExportError(Exception):
    """Raised for any error during CNN ONNX export."""

    pass


class CNNBuilder:
    """
    Build CNN models from an RL-sampled layer configuration.
    """

    def __init__(self, layer_config, in_channels=1, input_size=(28, 28), num_classes=26):
        """
        Args:
            layer_config: list of dicts, each describing a layer (conv, pool, linear)
            in_channels: input channels (1=grayscale, 3=RGB)
            input_size: spatial size of input images (height, width)
            num_classes: number of output classes (26 letters)
        """
        self.layer_config = layer_config
        self.in_channels = in_channels
        self.input_size = input_size
        self.num_classes = num_classes
        self.model = None

    def build(self):
        layers = []
        current_in_channels = self.in_channels
        h, w = self.input_size
        flatten_added = False

        for idx, cfg in enumerate(self.layer_config):
            t = cfg.get("layer_type")

            if t == "conv":
                layers.append(
                    nn.Conv2d(
                        in_channels=current_in_channels,
                        out_channels=cfg["out_channels"],
                        kernel_size=cfg["kernel_size"],
                        stride=cfg.get("stride", 1),
                        padding=cfg["kernel_size"] // 2,
                    )
                )
                act = cfg.get("activation", "none")
                if act.lower() != "none":
                    layers.append(ACTIVATIONS[act.lower()]())

                current_in_channels = cfg["out_channels"]
                # Update spatial size, because Linear layers later need the flattened feature map size, so we keep track of spatial dimensions dynamically.
                stride = cfg.get("stride", 1)
                h = (h + 2 * (cfg["kernel_size"] // 2) - cfg["kernel_size"]) // stride + 1
                w = (w + 2 * (cfg["kernel_size"] // 2) - cfg["kernel_size"]) // stride + 1

            elif t == "pool":
                mode = cfg.get("pool_mode", "max")
                kernel = cfg["kernel_size"]
                stride = cfg.get("stride", kernel)  # default stride = kernel size

                if mode == "max":
                    layers.append(nn.MaxPool2d(kernel, stride))
                elif mode == "avg":
                    layers.append(nn.AvgPool2d(kernel, stride))
                else:
                    raise InvalidLayerConfigError(f"Unknown pool mode '{mode}' at index {idx}")

                # Update spatial size
                h = (h - kernel) // stride + 1
                w = (w - kernel) // stride + 1

            elif (
                t == "linear"
            ):  # this assumes that there will be not conv or pool layers after a linear layer. meaning flatening should only occur once.
                if not flatten_added:
                    layers.append(nn.Flatten())
                    flatten_added = True
                    # Compute flattened input size
                    in_features = current_in_channels * h * w
                else:
                    in_features = layers[-1].out_features

                layers.append(nn.Linear(in_features, cfg["linear_units"]))
                act = cfg.get("activation", "none")
                if act.lower() != "none":
                    layers.append(ACTIVATIONS[act.lower()]())

            else:
                raise InvalidLayerConfigError(f"Unknown layer type '{t}' at index {idx}")

        # Final classifier, ensures that the features are flatened at the end if the layer_config does not include a linear layer at the end.
        if not flatten_added:
            layers.append(nn.Flatten())
            in_features = current_in_channels * h * w
        else:
            in_features = layers[-1].out_features

        layers.append(nn.Linear(in_features, self.num_classes))

        self.model = nn.Sequential(*layers)
        return self.model

    def export_to_onnx(self, input_size=(1, 28, 28), filename=None, opset=17):
        """
        Export the built CNN to ONNX format and save it a seperate file.

        Note, ONNX just mirrors the PyTorch model at the time of export.

        Args:
            input_size: tuple (C, H, W) of the input image (excluding batch size)
            filename: ONNX file name
            opset: ONNX opset version
        """
        if self.model is None:
            raise CNNExportError("Build the model first with .build() before exporting.")

        if not isinstance(input_size, tuple):
            raise CNNExportError("input_size must be a tuple (C, H, W)")

        if filename is None:
            # Auto-generate filename
            filename = f"cnn_model_{id(self)}.onnx"

        os.makedirs("saved_models", exist_ok=True)
        full_path = os.path.join("saved_models", filename)

        dummy_input = torch.randn(1, *input_size)
        torch.onnx.export(
            self.model,
            (dummy_input,),
            full_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=opset,
        )

        # Verify ONNX model
        onnx_model = onnx.load(filename)
        onnx.checker.check_model(onnx_model)
        return full_path
