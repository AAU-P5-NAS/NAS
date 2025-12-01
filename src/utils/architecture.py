from torch import Tensor
import torch.nn as nn
from src.utils.network_config import (
    SINGLE_LAYER_OBSERVATION_SIZE,
    NetworkConfig,
    update_spatial_dims,
)
from src.utils.layer_config import (
    LayerConfig,
    LayerType,
    PoolMode,
)

DROPOUT_PROBABILITY = 0.2


class Architecture(nn.Module):
    """
    An instantiated neural network based on a NetworkConfig.

    """

    def __init__(
        self, net_config: NetworkConfig, num_classes: int, input_dimensions: tuple[int, int, int]
    ):
        super().__init__()
        self.net_config = net_config
        self.num_classes = num_classes
        self.input_dimensions = input_dimensions
        self.apply_batch = (
            sum(1 for layer in self.net_config.layers if layer.layer_type == LayerType.CONV) >= 4
        )
        self.flattened = False
        self.model = self.build()

    def build(self) -> nn.Sequential:
        layers = []
        in_channels, h, w = self.input_dimensions

        for layer_config in self.net_config.layers:
            if layer_config.layer_type is LayerType.NONE:
                continue
            module, in_channels, h, w = self.build_single_layer(layer_config, in_channels, h, w)
            layers.extend(module)

        if not self.flattened:
            layers.append(nn.Flatten())
            layers.append(nn.Dropout(DROPOUT_PROBABILITY))
            in_channels = in_channels * h * w

        if in_channels != self.num_classes:
            layers.append(nn.Linear(in_channels, self.num_classes))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def build_single_layer(self, layer_config: LayerConfig, in_channels: int, h: int, w: int):
        """Builds one layer from LayerConfig and returns:
        - pytorch sequential (nn.sequential)
        - updated (in_channels, h, w)
        """
        module = []

        if layer_config.layer_type == LayerType.CONV:
            stride = layer_config.stride.to_stride()
            kernel = layer_config.kernel_size.to_kernel()
            padding = kernel // 2
            out_ch = layer_config.out_channels.to_channels()

            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_ch,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            )
            module.append(conv)

            # Add batch norm if specified
            if self.apply_batch:
                module.append(nn.BatchNorm2d(out_ch))

            # Add activation
            if layer_config.activation is not None:
                module.append(layer_config.activation.to_module())

            # Update dimensions
            h, w = update_spatial_dims(h, w, kernel, stride, padding)
            return module, out_ch, h, w

        elif layer_config.layer_type == LayerType.POOL:
            kernel = layer_config.kernel_size.to_kernel()
            stride = layer_config.stride.to_stride()

            if layer_config.pool_mode is PoolMode.MAX:
                padding = 0
                if w < kernel:
                    padding = kernel // 2
                pool = nn.MaxPool2d(kernel, stride, padding)
            else:
                # pool = nn.AvgPool2d(kernel, stride)
                raise ValueError("avg pool mode cannot be added anymore")

            module.append(pool)

            if layer_config.activation is not None:
                module.append(layer_config.activation.to_module())

            h, w = update_spatial_dims(h, w, kernel, stride)
            return module, in_channels, h, w

        elif layer_config.layer_type == LayerType.LINEAR:
            if not self.flattened:
                module.append(nn.Flatten())
                module.append(nn.Dropout(DROPOUT_PROBABILITY))
                self.flattened = True

                in_channels = in_channels * h * w
                h, w = 1, 1

            out_units = layer_config.linear_units.to_units()

            if out_units is None:
                raise ValueError("Linear layer must define number of units")

            fc = nn.Linear(in_channels, out_units)
            module.append(fc)
            if layer_config.activation is not None:
                module.append(layer_config.activation.to_module())
            return module, out_units, 1, 1

        else:
            raise ValueError("layer type not supported")


def flatten_cnn_config(
    rlconfig: NetworkConfig, max_layers: int, padded_with_zeros: bool = True
) -> list[int]:
    # each layer has 8 slots, 0 for unused slots (none values for enums)
    flat_config = []
    for index, layer in enumerate(rlconfig.layers):
        data = layer.model_dump()
        flat_layer_config = []
        for _, item in data.items():
            flat_layer_config.append(item.value)

        if len(flat_layer_config) != SINGLE_LAYER_OBSERVATION_SIZE:
            raise ValueError("Layer config size mismatch")

        flat_config.extend(flat_layer_config)
    if max_layers * SINGLE_LAYER_OBSERVATION_SIZE < len(flat_config):
        raise ValueError("flat_config is larger than allowed size")

    if padded_with_zeros:
        flat_config.extend((max_layers * SINGLE_LAYER_OBSERVATION_SIZE - len(flat_config)) * [0])

    return flat_config
