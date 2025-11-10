from typing import Dict, cast
from torch import Tensor
import torch.nn as nn
from utils.network_utils import LayerConfig, LayerType, NetworkConfig, PoolMode, update_spatial_dims


class GraphCnn(nn.Module):
    def __init__(
        self, net_config: NetworkConfig, num_classes: int, input_dimensions: tuple[int, int, int]
    ):
        super().__init__()
        self.net_config = net_config
        self.num_classes = num_classes
        self.input_dimensions = input_dimensions
        self.layers = nn.ModuleList()
        self.projections = nn.ModuleDict()
        self.build()

    def build(self):
        in_channels, h, w = self.input_dimensions
        has_flattened = False
        for index, layer_config in enumerate(self.net_config.layers):
            if layer_config.layer_type == LayerType.LINEAR and not has_flattened:
                self.layers.append(nn.Flatten())
                has_flattened = True
                in_channels *= h * w
                h, w = 1, 1  # no longer relevant

            if layer_config.layer_type is LayerType.NONE:
                continue

            layer, out_channels, h, w = build_single_layer(layer_config, in_channels, h, w)
            self.layers.append(layer)

            skip_from = layer_config.skip_connection
            if skip_from is not None:
                self._add_skip_projection(skip_from, index, out_channels, h, w)

            in_channels = out_channels

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
        """Add a projection layer to match dimensions for skip connections.
        Cases:
            1. Conv/Pool → Conv/Pool: project only if in/out channels differ
            2. Conv/Pool → Linear: project using flatten + linear (if needed)
            3. Linear → Linear: project only if feature dims differ
        """
        proj_name = f"from_{skip_from}_to_{skip_to}"
        from_layer = self.layers[skip_from]
        in_channels = self._infer_out_channels(from_layer)

        is_to_conv = h > 1 and w > 1

        if is_to_conv and in_channels != out_channels:  # case 1
            self.projections[proj_name] = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        elif not is_to_conv and isinstance(from_layer, (nn.Conv2d, nn.Sequential)):  # case 2
            flat_features = in_channels * h * w
            if flat_features == out_channels:
                self.projections[proj_name] = nn.Flatten()
            else:
                self.projections[proj_name] = nn.Sequential(
                    nn.Flatten(), nn.Linear(flat_features, out_channels)
                )
        else:  # case 3
            if in_channels != out_channels:
                self.projections[proj_name] = nn.Linear(in_channels, out_channels)


def build_single_layer(layer_config: LayerConfig, in_channels: int, h: int, w: int):
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
