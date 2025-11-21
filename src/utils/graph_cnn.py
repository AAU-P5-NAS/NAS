from typing import Dict, cast
from torch import Tensor
import torch.nn as nn
from src.utils.network_utils import (
    LayerConfig,
    LayerType,
    NetworkConfig,
    PoolMode,
    update_spatial_dims,
)

DROPOUT_PROBABILITY = 0.2


class GraphCnn(nn.Module):
    def __init__(
        self, net_config: NetworkConfig, num_classes: int, input_dimensions: tuple[int, int, int]
    ):
        super().__init__()
        self.net_config = net_config
        self.num_classes = num_classes
        self.input_dimensions = input_dimensions
        self.layers = nn.ModuleList()
        self.layer_shapes: dict[int, tuple[int, int, int]] = {}
        self.projections = nn.ModuleDict()
        self.has_flattened = False
        self.build()

    def build(self):
        in_channels, h, w = self.input_dimensions
        apply_batch = (
            sum(1 for layer in self.net_config.layers if layer.layer_type == LayerType.CONV) >= 4
        )
        for index, layer_config in enumerate(self.net_config.layers):
            if layer_config.layer_type is LayerType.NONE:
                continue

            layer, out_channels, h, w = self.build_single_layer(
                layer_config, in_channels, h, w, apply_batch
            )
            self.layers.append(layer)
            self.layer_shapes[index] = (out_channels, h, w)

            skip_from = layer_config.skip_connection
            if skip_from is not None:
                self._add_skip_projection(skip_from, index, out_channels, h, w)

            in_channels = out_channels

        if not self.has_flattened:
            self.layers.append(nn.Flatten())
            self.layers.append(nn.Dropout(DROPOUT_PROBABILITY))
            in_channels *= h * w
            h, w = 1, 1

        if in_channels != self.num_classes:
            self.layers.append(nn.Linear(in_channels, self.num_classes))

    def forward(self, x: Tensor) -> Tensor:
        outputs: Dict[int, Tensor] = {}
        for index, layer in enumerate(self.layers):
            x = layer(x)

            for proj_name, proj_layer in self.projections.items():
                if f"_to_{index}" in proj_name:
                    skip_from = int(proj_name.split("_")[1])
                    skip_out = outputs[skip_from]
                    skip_proj = proj_layer(skip_out)
                    x = x + skip_proj
            outputs[index] = x
        return x

    def _infer_out_channels(self, layer: nn.Module) -> int:
        """Try to infer the output dimensionality (channels or features)."""
        if hasattr(layer, "out_channels"):
            return cast(int, layer.out_channels)
        if isinstance(layer, nn.Sequential):
            for sub in reversed(layer):
                if hasattr(sub, "out_channels"):
                    return cast(int, sub.out_channels)
                elif isinstance(sub, nn.Linear):
                    return sub.out_features
        if isinstance(layer, nn.Linear):
            return layer.out_features
        raise AttributeError("Cannot infer output channels")

    def _add_skip_projection(self, skip_from: int, skip_to: int, out_channels: int, h: int, w: int):
        """
        Add a projection layer to match dimensions for skip connections.

        Cases:
            1. Conv/Pool -> Conv/Pool: project only if in/out channels differ
            2. Conv/Pool -> Linear: flatten + optional linear projection
            3. Linear -> Linear: project only if feature dims differ
        """
        proj_name = f"from_{skip_from}_to_{skip_to}"

        if skip_from in self.layer_shapes:
            in_channels, h_from, w_from = self.layer_shapes[skip_from]
        else:
            in_channels, h_from, w_from = self.input_dimensions

        is_to_conv = h > 1 and w > 1
        is_from_conv = h_from > 1 and w_from > 1

        # Case 1:
        if is_from_conv and is_to_conv:
            if in_channels != out_channels:
                self.projections[proj_name] = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Case 2
        elif is_from_conv and not is_to_conv:
            flat_features = in_channels * h_from * w_from
            if flat_features == out_channels:
                self.projections[proj_name] = nn.Flatten()
            else:
                self.projections[proj_name] = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(flat_features, out_channels),
                )

        # Case 3
        else:
            if in_channels != out_channels:
                self.projections[proj_name] = nn.Linear(in_channels, out_channels)

    def build_single_layer(
        self, layer_config: LayerConfig, in_channels: int, h: int, w: int, apply_batch: bool = True
    ):
        """Builds one layer from LayerConfig and returns:
        - pytorch module (nn.module)
        - updated (in_channels, h, w)
        """
        modules = []

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
            modules.append(conv)

            # Add batch norm if specified
            if apply_batch:
                modules.append(nn.BatchNorm2d(out_ch))

            # Add activation
            if layer_config.activation is not None:
                modules.append(layer_config.activation.to_module())

            # Update dimensions
            h, w = update_spatial_dims(h, w, kernel, stride, padding)
            return nn.Sequential(*modules), out_ch, h, w

        elif layer_config.layer_type == LayerType.POOL:
            kernel = layer_config.kernel_size.to_kernel()
            stride = layer_config.stride.to_stride()

            if layer_config.pool_mode is PoolMode.MAX:
                pool = nn.MaxPool2d(kernel, stride)
            else:
                pool = nn.AvgPool2d(kernel, stride)

            modules.append(pool)
            h, w = update_spatial_dims(h, w, kernel, stride)
            return nn.Sequential(*modules), in_channels, h, w
        elif layer_config.layer_type == LayerType.LINEAR:
            if not self.has_flattened:
                modules.append(nn.Flatten())
                modules.append(nn.Dropout(DROPOUT_PROBABILITY))
                self.has_flattened = True

                in_channels = in_channels * h * w
                h, w = 1, 1

            out_units = layer_config.linear_units.to_units()

            if out_units is None:
                raise ValueError("Linear layer must define number of units")

            fc = nn.Linear(in_channels, out_units)
            modules.append(fc)
            if layer_config.activation is not None:
                modules.append(layer_config.activation.to_module())
            return nn.Sequential(*modules), out_units, 1, 1

        else:
            raise ValueError("layer type not supported")
