from typing import Optional
import numpy as np
from src.action_masking.action_masking import sample_actions
from src.action_masking.action_masking_utils import (
    MaskContext,
    get_logit_slices,
    standard_stochastic_sampling,
)
from src.utils.cnn_builder import flatten_cnn_config
from src.utils.network_utils import (
    LayerConfig,
    NetworkConfig,
    EMPTY_DECISIONS,
    StandardAction,
    LayerType,
    OutChannels,
    KernelSize,
    Stride,
    LinearUnits,
    PoolMode,
    ActivationFunction,
)


def get_obs(network_config: NetworkConfig, max_layers: int) -> np.ndarray:
    flattened_obs = flatten_cnn_config(network_config, max_layers)
    return np.array(flattened_obs, dtype=np.float32)


def get_logits(
    next_action: Optional[StandardAction] = None, next_layer: Optional[LayerType] = None
):
    # Return logits initialized with standard random values (e.g., normal distribution)
    sum = (
        len(StandardAction)
        + len(LayerType)
        + len(OutChannels)
        + len(KernelSize)
        + len(Stride)
        + len(LinearUnits)
        + len(PoolMode)
        + len(ActivationFunction)
    )
    logits = np.random.randn(sum).astype(np.float32)

    if next_action is not None:
        logits[0 : len(StandardAction)] = -np.inf
        logits[next_action.value] = 1
        if next_action == StandardAction.NONE:
            logits[len(StandardAction) : len(StandardAction) + len(LayerType)] = -np.inf
            logits[len(StandardAction) + LayerType.NONE.value] = 1

    if next_layer is not None:
        start_idx = len(StandardAction)
        end_idx = start_idx + len(LayerType)
        logits[start_idx:end_idx] = -np.inf
        logits[start_idx + next_layer.value] = 1

    return logits


def test_first_action_is_add_layer():
    empty_config = NetworkConfig(layers=[])

    ctx = MaskContext(
        logits=get_logits(),
        observation=get_obs(empty_config, max_layers=20),
        slices=get_logit_slices(),
        sampling_strategy=standard_stochastic_sampling,
        max_layers=20,
        decisions=EMPTY_DECISIONS,
        input_dimensions=(3, 32, 32),
    )

    decisions, masked_logits = sample_actions(ctx)

    assert decisions.action_choice == StandardAction.ADD_LAYER


def test_max_layers():
    MAX_LAYERS = 20
    full_config = NetworkConfig(
        layers=[
            LayerConfig(
                layer_type=LayerType.CONV,
                out_channels=OutChannels.CH_16,
                kernel_size=KernelSize.KS_3,
                stride=Stride.S_1,
                activation=ActivationFunction.RELU,
            )
            for _ in range(MAX_LAYERS)
        ]
    )

    ctx = MaskContext(
        logits=get_logits(),
        observation=get_obs(full_config, max_layers=MAX_LAYERS),
        slices=get_logit_slices(),
        sampling_strategy=standard_stochastic_sampling,
        max_layers=MAX_LAYERS,
        decisions=EMPTY_DECISIONS,
        input_dimensions=(3, 32, 32),
    )

    decisions, masked_logits = sample_actions(ctx)

    assert decisions.action_choice == StandardAction.NONE


def test_after_linear_no_conv_or_pool():
    MAX_LAYERS = 20
    config = NetworkConfig(
        layers=[
            LayerConfig(
                layer_type=LayerType.CONV,
                out_channels=OutChannels.CH_16,
                kernel_size=KernelSize.KS_3,
                stride=Stride.S_1,
                activation=ActivationFunction.RELU,
            ),
            LayerConfig(
                layer_type=LayerType.LINEAR,
                linear_units=LinearUnits.LU_64,
                activation=ActivationFunction.TANH,
            ),
        ]
    )

    ctx = MaskContext(
        logits=get_logits(),
        observation=get_obs(config, max_layers=MAX_LAYERS),
        slices=get_logit_slices(),
        sampling_strategy=standard_stochastic_sampling,
        max_layers=MAX_LAYERS,
        decisions=EMPTY_DECISIONS,
        input_dimensions=(3, 32, 32),
    )

    decisions, masked_logits = sample_actions(ctx)

    assert decisions.layer_type_choice != LayerType.CONV
    assert decisions.layer_type_choice != LayerType.POOL


def test_layertype_is_none_after_none_action():
    MAX_LAYERS = 20
    config = NetworkConfig(
        layers=[
            LayerConfig(
                layer_type=LayerType.CONV,
                out_channels=OutChannels.CH_16,
                kernel_size=KernelSize.KS_3,
                stride=Stride.S_1,
                activation=ActivationFunction.RELU,
            ),
        ]
    )

    ctx = MaskContext(
        logits=get_logits(next_action=StandardAction.NONE),
        observation=get_obs(config, max_layers=MAX_LAYERS),
        slices=get_logit_slices(),
        sampling_strategy=standard_stochastic_sampling,
        max_layers=MAX_LAYERS,
        decisions=EMPTY_DECISIONS,
        input_dimensions=(3, 32, 32),
    )

    decisions, masked_logits = sample_actions(ctx)
    print("Decisions:", decisions)

    assert decisions.layer_type_choice == LayerType.NONE


def test_no_outputchannels_chosen_after_linear():
    MAX_LAYERS = 20
    config = NetworkConfig(
        layers=[
            LayerConfig(
                layer_type=LayerType.CONV,
                out_channels=OutChannels.CH_16,
                kernel_size=KernelSize.KS_3,
                stride=Stride.S_1,
                activation=ActivationFunction.RELU,
            ),
            LayerConfig(
                layer_type=LayerType.LINEAR,
                linear_units=LinearUnits.LU_64,
                activation=ActivationFunction.TANH,
            ),
        ]
    )

    ctx = MaskContext(
        logits=get_logits(),
        observation=get_obs(config, max_layers=MAX_LAYERS),
        slices=get_logit_slices(),
        sampling_strategy=standard_stochastic_sampling,
        max_layers=MAX_LAYERS,
        decisions=EMPTY_DECISIONS,
        input_dimensions=(3, 32, 32),
    )

    decisions, masked_logits = sample_actions(ctx)

    assert decisions.out_channels_choice == OutChannels.NONE


def test_correct_kernel_size_1():
    MAX_LAYERS = 20
    config = NetworkConfig(
        layers=[
            LayerConfig(
                layer_type=LayerType.POOL,
                pool_mode=PoolMode.MAX,
                kernel_size=KernelSize.KS_3,
                stride=Stride.S_2,
                activation=ActivationFunction.RELU,
            ),
            LayerConfig(
                layer_type=LayerType.POOL,
                pool_mode=PoolMode.MAX,
                kernel_size=KernelSize.KS_3,
                stride=Stride.S_2,
                activation=ActivationFunction.RELU,
            ),
            LayerConfig(
                layer_type=LayerType.POOL,
                pool_mode=PoolMode.MAX,
                kernel_size=KernelSize.KS_3,
                stride=Stride.S_2,
                activation=ActivationFunction.RELU,
            ),
            LayerConfig(
                layer_type=LayerType.POOL,
                pool_mode=PoolMode.MAX,
                kernel_size=KernelSize.KS_3,
                stride=Stride.S_2,
                activation=ActivationFunction.RELU,
            ),
            LayerConfig(
                layer_type=LayerType.CONV,
                out_channels=OutChannels.CH_16,
                kernel_size=KernelSize.KS_1,
                stride=Stride.S_1,
                activation=ActivationFunction.RELU,
            ),
        ]
    )

    ctx = MaskContext(
        logits=get_logits(next_action=StandardAction.ADD_LAYER, next_layer=LayerType.CONV),
        observation=get_obs(config, max_layers=MAX_LAYERS),
        slices=get_logit_slices(),
        sampling_strategy=standard_stochastic_sampling,
        max_layers=MAX_LAYERS,
        decisions=EMPTY_DECISIONS,
        input_dimensions=(3, 32, 32),
    )

    decisions, masked_logits = sample_actions(ctx)

    assert decisions.layer_type_choice == LayerType.CONV
    assert decisions.kernel_size_choice == KernelSize.KS_1


def test_correct_kernel_size_1or3():
    MAX_LAYERS = 20
    config = NetworkConfig(
        layers=[
            LayerConfig(
                layer_type=LayerType.POOL,
                pool_mode=PoolMode.MAX,
                kernel_size=KernelSize.KS_3,
                stride=Stride.S_2,
                activation=ActivationFunction.RELU,
            ),
            LayerConfig(
                layer_type=LayerType.POOL,
                pool_mode=PoolMode.MAX,
                kernel_size=KernelSize.KS_3,
                stride=Stride.S_2,
                activation=ActivationFunction.RELU,
            ),
            LayerConfig(
                layer_type=LayerType.POOL,
                pool_mode=PoolMode.MAX,
                kernel_size=KernelSize.KS_3,
                stride=Stride.S_2,
                activation=ActivationFunction.RELU,
            ),
            LayerConfig(
                layer_type=LayerType.CONV,
                out_channels=OutChannels.CH_16,
                kernel_size=KernelSize.KS_1,
                stride=Stride.S_1,
                activation=ActivationFunction.RELU,
            ),
        ]
    )

    ctx = MaskContext(
        logits=get_logits(next_action=StandardAction.ADD_LAYER, next_layer=LayerType.CONV),
        observation=get_obs(config, max_layers=MAX_LAYERS),
        slices=get_logit_slices(),
        sampling_strategy=standard_stochastic_sampling,
        max_layers=MAX_LAYERS,
        decisions=EMPTY_DECISIONS,
        input_dimensions=(3, 32, 32),
    )

    decisions, masked_logits = sample_actions(ctx)

    assert decisions.layer_type_choice == LayerType.CONV
    assert (
        decisions.kernel_size_choice == KernelSize.KS_3
        or decisions.kernel_size_choice == KernelSize.KS_1
    )


def test_correct_stride_after_kernelsize():
    pass
