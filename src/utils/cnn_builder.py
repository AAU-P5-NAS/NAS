import torch.nn as nn
import torch.onnx
import onnx
import os
import warnings
from typing import Optional, Tuple

from src.utils.network_utils import (
    ActivationFunction,
    CNNExportError,
    KernelSize,
    LayerConfig,
    LayerType,
    LinearUnits,
    NetworkConfig,
    OutChannels,
    PoolMode,
    update_spatial_dims,
)


class CNNBuilder:
    def __init__(
        self,
        rl_config: NetworkConfig,
        num_classes: int,
        input_size: Tuple[int, int] = (28, 28),
    ):
        self.rl_config = rl_config
        self.input_size = input_size
        self.num_classes = num_classes
        self.model: Optional[nn.Sequential] = None

    def build(self):
        layers = []
        current_in_channels = 1
        h, w = self.input_size
        conv_pool_layers = [
            layer
            for layer in self.rl_config.layers
            if layer.layer_type in (LayerType.CONV, LayerType.POOL)
        ]
        linear_layers = [
            layer for layer in self.rl_config.layers if layer.layer_type is LayerType.LINEAR
        ]

        for layer in conv_pool_layers:
            if layer.layer_type is LayerType.CONV:
                stride = layer.stride.to_stride() if layer.stride is not None else 1
                assert layer.kernel_size is not None
                kernel = layer.kernel_size.to_kernel()
                assert kernel is not None  #
                padding = kernel // 2
                assert layer.out_channels is not None
                out_ch = layer.out_channels.to_channels()
                assert out_ch is not None
                assert current_in_channels is not None
                layers.append(
                    nn.Conv2d(
                        in_channels=current_in_channels,
                        out_channels=out_ch,
                        kernel_size=kernel,
                        stride=stride,
                        padding=padding,
                    )
                )
                assert layer.activation is not None
                layers.append(layer.activation.to_module())
                current_in_channels = out_ch
                h, w = update_spatial_dims(h, w, kernel, stride, padding)

            elif layer.layer_type is LayerType.POOL:
                assert layer.kernel_size is not None
                stride = (
                    layer.stride.to_stride()
                    if layer.stride is not None
                    else layer.kernel_size.to_kernel()
                )
                kernel = layer.kernel_size.to_kernel()
                assert kernel is not None
                if layer.pool_mode is PoolMode.MAX:
                    layers.append(nn.MaxPool2d(kernel, stride))
                else:
                    layers.append(nn.AvgPool2d(kernel, stride))

                assert stride is not None
                h, w = update_spatial_dims(h, w, kernel, stride)

        layers.append(nn.Flatten())
        assert current_in_channels is not None

        in_features = current_in_channels * h * w

        for layer in linear_layers:
            assert in_features is not None
            assert layer.linear_units is not None
            layers.append(nn.Linear(in_features, layer.linear_units.to_units()))
            assert layer.activation is not None
            layers.append(layer.activation.to_module())
            in_features = layer.linear_units.to_units()

        layers.append(nn.Linear(in_features, self.num_classes))

        self.model = nn.Sequential(*layers)
        return self.model

    def export_to_onnx(self, save_in_seperate_file: bool = False, opset=17):
        """
        Export the built CNN to ONNX format and save it a seperate file.
        Note, ONNX just mirrors the PyTorch model at the time of export.

        """

        if self.model is None:
            raise CNNExportError("Build the model first with .build() before exporting.")

        if save_in_seperate_file is True:
            # Auto-generate filename
            filename = f"cnn_model_{id(self)}.onnx"
        else:
            filename = "cnn.onnx"

        input_size = (1, 28, 28)

        os.makedirs("saved_models", exist_ok=True)
        full_path = os.path.join("saved_models", filename)
        dummy_input = torch.randn(1, *input_size)
        # Suppress deprecation warnings for ONNX export. smthn about a new version of onnx exporter, but the current one still work
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            torch.onnx.export(
                self.model,
                (dummy_input,),  # tuple of inputs
                full_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                opset_version=opset,
                export_params=True,
                training=torch.onnx.TrainingMode.EVAL,  # export in inference mode
            )

        # Verify ONNX model
        onnx_model = onnx.load(full_path)
        onnx.checker.check_model(onnx_model)
        return full_path


def flatten_cnn_config(rlconfig: NetworkConfig, max_layers: int) -> list[int]:
    # each layer has 7 slots, 0 for  unused slots (none values for enums)
    flat_config = []
    for layer in rlconfig.layers:
        data = layer.model_dump()
        flat_layer_config = []
        for key, item in data.items():
            if item is not None:
                flat_layer_config.append(item.value)
            else:
                flat_layer_config.append(0)
        while len(flat_layer_config) < 7:
            flat_layer_config.append(0)

        flat_config.extend(flat_layer_config)
    if max_layers * 7 < len(flat_config):
        raise ValueError("flat_config is larger than allowed size")
    flat_config.extend((max_layers * 7 - len(flat_config)) * [0])

    return flat_config


# ------------------------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    # Define a sample RL-generated config
    config = NetworkConfig(
        layers=[
            LayerConfig(
                layer_type=LayerType.CONV,
                out_channels=OutChannels.CH_16,
                kernel_size=KernelSize.KS_3,
                activation=ActivationFunction.RELU,
            ),
            LayerConfig(
                layer_type=LayerType.POOL, pool_mode=PoolMode.MAX, kernel_size=KernelSize.KS_1
            ),
            LayerConfig(
                layer_type=LayerType.LINEAR,
                linear_units=LinearUnits.LU_64,
                activation=ActivationFunction.TANH,
            ),
        ]
    )

    # Instantiate the CNN builder
    cnn_builder = CNNBuilder(rl_config=config, input_size=(28, 28), num_classes=26)

    # Build the PyTorch model
    model = cnn_builder.build()

    # Print the model architecture

    # Optional: export to ONNX

    actions = flatten_cnn_config(config, 4)
