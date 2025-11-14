import numpy as np
import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.action_masking.action_masking_utils import (
    NO_ACTION_DECISIONS,
    MaskContext,
    sample_action_for_slice,
    sample_skip_connection,
)
from src.utils.network_utils import (
    SINGLE_LAYER_OBSERVATION_SIZE,
    StandardAction,
    LayerType,
    OutChannels,
    KernelSize,
    Stride,
    LinearUnits,
    PoolMode,
    ActivationFunction,
    calculate_output_dimensions,
    get_latest_layer,
    get_layer_from_index,
    get_output_dimensions,
    get_valid_kernel_sizes,
    get_valid_strides,
)


def sample_actions(ctx: MaskContext):
    ctx.observation = ctx.observation.flatten()
    ctx.logits = mask_action_type_sequential(ctx)
    ctx.decisions.action_choice = sample_action_for_slice(ctx, StandardAction, "standard_actions")
    if ctx.decisions.action_choice == StandardAction.NONE:
        masked_logits = ctx.logits.copy()
        masked_logits[:] = -np.inf
        masked_logits[ctx.slices.standard_actions[StandardAction.NONE]] = 1
        return NO_ACTION_DECISIONS, masked_logits

    ctx.logits = mask_layer_type_sequential(ctx)
    ctx.decisions.layer_type_choice = sample_action_for_slice(ctx, LayerType, "layer_type")
    ctx.logits = mask_out_channels_sequential(ctx)
    ctx.decisions.out_channels_choice = sample_action_for_slice(ctx, OutChannels, "out_channels")
    ctx.logits = mask_kernel_size_sequential(ctx)
    ctx.decisions.kernel_size_choice = sample_action_for_slice(ctx, KernelSize, "kernel_size")
    ctx.logits = mask_stride_sequential(ctx)
    ctx.decisions.stride_choice = sample_action_for_slice(ctx, Stride, "stride")
    ctx.logits = mask_linear_units_sequential(ctx)
    ctx.decisions.linear_units_choice = sample_action_for_slice(ctx, LinearUnits, "linear_units")
    ctx.logits = mask_pool_mode_sequential(ctx)
    ctx.decisions.pool_mode_choice = sample_action_for_slice(ctx, PoolMode, "pool_mode")
    ctx.logits = mask_activation_function_sequential(ctx)
    ctx.decisions.activation_function_choice = sample_action_for_slice(
        ctx, ActivationFunction, "activation_function"
    )
    ctx.logits = mask_skip_connection_sequential(ctx)
    ctx.decisions.skip_connection_choice = sample_skip_connection(ctx)

    return ctx.decisions, ctx.logits


def mask_action_type_sequential(ctx: MaskContext):
    """Mask action types based on current observation and strategy. Raises MaxLayersReachedException if max layers reached."""
    new_logits = ctx.logits.copy()

    if ctx.action_count >= ctx.max_layers:
        # Max layers reached, can only choose NONE
        new_logits[ctx.slices.standard_actions.all] = -np.inf
        new_logits[ctx.slices.standard_actions[StandardAction.NONE]] = 1
        return new_logits

    if ctx.action_count == 0:
        # No layers yet, can only add
        new_logits[ctx.slices.standard_actions.all] = -np.inf
        new_logits[ctx.slices.standard_actions[StandardAction.ADD_LAYER]] = 1
        return new_logits

    return new_logits


def mask_layer_type_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()
    print("logits for layer type before masking:", new_logits[ctx.slices.layer_type.all])
    print("action choice:", ctx.decisions.action_choice)
    if ctx.decisions.action_choice == StandardAction.NONE:
        new_logits[ctx.slices.layer_type.all] = -np.inf
        new_logits[ctx.slices.layer_type[LayerType.NONE]] = 1
        print("logits for layer type after masking NONE:", new_logits[ctx.slices.layer_type.all])
        return new_logits

    linear_layer_exists = any(
        ctx.observation[i] == LayerType.LINEAR.value
        for i in range(0, len(ctx.observation), SINGLE_LAYER_OBSERVATION_SIZE)
    )
    if linear_layer_exists:
        new_logits[ctx.slices.layer_type.all] = -np.inf
        new_logits[ctx.slices.layer_type[LayerType.LINEAR]] = 1  # only linear is valid
        return new_logits

    previous_layer = get_latest_layer(ctx.observation, ctx.action_count, ctx.max_layers)
    if previous_layer is None or previous_layer.layer_type != LayerType.CONV:
        # if no previous layer or previous layer is not conv, cannot add pool (typically pool follows conv)
        new_logits[ctx.slices.layer_type[LayerType.POOL]] = -np.inf

    new_logits[
        ctx.slices.layer_type[LayerType.NONE]
    ] = -np.inf  # NONE is not valid when adding a layer

    return new_logits


def mask_out_channels_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()

    if (
        ctx.decisions.layer_type_choice == LayerType.LINEAR
        or ctx.decisions.layer_type_choice == LayerType.NONE
    ):
        new_logits[ctx.slices.out_channels.all] = -np.inf
        new_logits[ctx.slices.out_channels[OutChannels.NONE]] = 1
        return new_logits

    new_logits[ctx.slices.out_channels[OutChannels.NONE]] = -np.inf  # NONE is not valid when adding
    return new_logits


def mask_kernel_size_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()

    if (
        ctx.decisions.layer_type_choice == LayerType.NONE
        or ctx.decisions.layer_type_choice == LayerType.LINEAR
    ):
        # mask all kernel sizes
        new_logits[ctx.slices.kernel_size.all] = -np.inf
        new_logits[ctx.slices.kernel_size[KernelSize.NONE]] = 1
        return new_logits

    latest_output_dims = get_output_dimensions(
        ctx.observation, ctx.input_dimensions[1:], ctx.max_layers
    )

    if ctx.decisions.layer_type_choice == LayerType.POOL:
        valid_kernels = get_valid_kernel_sizes(latest_output_dims, padding=0)
    else:
        valid_kernels = get_valid_kernel_sizes(latest_output_dims, padding=1)

    invalid_kernels = [k for k in KernelSize if k not in valid_kernels]
    for kernel in invalid_kernels:
        invalid_kernel_index = ctx.slices.kernel_size[kernel.value]
        new_logits[invalid_kernel_index] = -np.inf

    new_logits[ctx.slices.kernel_size[KernelSize.NONE]] = -np.inf
    return new_logits


def mask_stride_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()

    if (
        ctx.decisions.layer_type_choice == LayerType.NONE
        or ctx.decisions.layer_type_choice == LayerType.LINEAR
    ):
        new_logits[ctx.slices.stride.all] = -np.inf
        new_logits[ctx.slices.stride[Stride.NONE]] = 1
        return new_logits

    kernel_size_chosen = ctx.decisions.kernel_size_choice
    latest_output_dims = get_output_dimensions(
        ctx.observation, ctx.input_dimensions[1:], ctx.max_layers
    )
    if ctx.decisions.layer_type_choice == LayerType.POOL:
        valid_strides = get_valid_strides(latest_output_dims, kernel_size_chosen, padding=0)
    else:
        valid_strides = get_valid_strides(latest_output_dims, kernel_size_chosen, padding=1)
    invalid_strides = [s for s in Stride if s not in valid_strides]

    for stride in invalid_strides:
        invalid_stride_index = ctx.slices.stride[stride.value]
        new_logits[invalid_stride_index] = -np.inf

    new_logits[ctx.slices.stride[Stride.NONE]] = -np.inf

    return new_logits


def mask_linear_units_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()

    if ctx.decisions.layer_type_choice != LayerType.LINEAR:
        new_logits[ctx.slices.linear_units.all] = -np.inf
        new_logits[ctx.slices.linear_units[LinearUnits.NONE]] = 1
        return new_logits

    new_logits[ctx.slices.linear_units[LinearUnits.NONE]] = -np.inf
    return new_logits


def mask_pool_mode_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()

    if ctx.decisions.layer_type_choice != LayerType.POOL:
        new_logits[ctx.slices.pool_mode.all] = -np.inf
        new_logits[ctx.slices.pool_mode[PoolMode.NONE]] = 1
        return new_logits

    new_logits[ctx.slices.pool_mode[PoolMode.NONE]] = -np.inf
    return new_logits


def mask_activation_function_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()
    # Mask NONE activation - not useful for hidden layers
    new_logits[ctx.slices.activation_function[ActivationFunction.NONE]] = -np.inf
    # Mask SOFTMAX activation - should only be used in output layer, not hidden layers
    new_logits[ctx.slices.activation_function[ActivationFunction.SOFTMAX]] = -np.inf

    return new_logits


def mask_skip_connection_sequential(ctx: MaskContext):
    new_logits = ctx.logits.copy()

    if ctx.action_count <= 2:
        # first two layers are not allowed to have skip connections from previous layers.
        new_logits[ctx.slices.skip_connection.all] = -np.inf
        new_logits[ctx.slices.skip_connection[ctx.max_layers - 1]] = (
            1  # no skip connection -> last layer index
        )
        return new_logits

    last_layer_output_dims = get_output_dimensions(
        ctx.observation, ctx.input_dimensions[1:], ctx.max_layers
    )
    input_dim = ctx.input_dimensions[1:]

    for i in range(0, ctx.action_count - 2):
        currentLayerConfig = get_layer_from_index(ctx.observation, i, ctx.max_layers)
        output_dim = calculate_output_dimensions(input_dim, currentLayerConfig)

        if output_dim != last_layer_output_dims:
            # incompatible dimensions for skip connection
            new_logits[ctx.slices.skip_connection[i]] = -np.inf
        input_dim = output_dim

    # mask previous layer, current layer and future layers
    for i in range(ctx.action_count - 2, ctx.max_layers):
        new_logits[ctx.slices.skip_connection[i]] = -np.inf

    # Always keep "no skip" valid
    new_logits[ctx.slices.skip_connection[ctx.max_layers - 1]] = 1.0
    return new_logits
