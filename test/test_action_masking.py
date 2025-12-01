from typing import Optional
import numpy as np
from src.agent.action_masking.action_masking import sample_actions
from src.agent.action_masking.action_masking_utils import (
    EMPTY_DECISIONS,
    MaskContext,
    get_logit_slices,
    standard_stochastic_sampling,
)
from src.utils.architecture import flatten_cnn_config
from src.utils.layer_config import (
    LayerConfig,
    LayerType,
    StandardAction,
    OutChannels,
    KernelSize,
    Stride,
    LinearUnits,
    PoolMode,
    ActivationFunction,
)
from src.utils.network_config import NetworkConfig


def get_obs(network_config: NetworkConfig, max_layers: int) -> np.ndarray:
    flattened_obs = flatten_cnn_config(network_config, max_layers)
    return np.array(flattened_obs, dtype=np.float32)


# Helper function
def get_logits(
    next_action: Optional[StandardAction] = None,
    next_layer: Optional[LayerType] = None,
    next_kernel_size: Optional[KernelSize] = None,
    next_stride: Optional[Stride] = None,
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
        + 20  # assuming max_layers=20 for skip connections
    )
    print(sum)
    logits = np.random.randn(sum).astype(np.float32)

    if next_action is not None:
        logits[0 : len(StandardAction)] = -np.inf
        logits[next_action.value] = 1

    if next_layer is not None:
        start_idx = len(StandardAction)
        end_idx = start_idx + len(LayerType)
        logits[start_idx:end_idx] = -np.inf
        logits[start_idx + next_layer.value] = 1

    if next_kernel_size is not None:
        start_idx = len(StandardAction) + len(LayerType) + len(OutChannels)
        end_idx = start_idx + len(KernelSize)
        logits[start_idx:end_idx] = -np.inf
        logits[start_idx + next_kernel_size.value] = 1

    if next_stride is not None:
        start_idx = len(StandardAction) + len(LayerType) + len(OutChannels) + len(KernelSize)
        end_idx = start_idx + len(Stride)
        logits[start_idx:end_idx] = -np.inf
        logits[start_idx + next_stride.value] = 1
    return logits


def test_first_action_is_add_layer():
    empty_config = NetworkConfig(layers=[])

    ctx = MaskContext(
        logits=get_logits(),
        observation=get_obs(empty_config, max_layers=20),
        slices=get_logit_slices(max_layers=20),
        sampling_strategy=standard_stochastic_sampling,
        max_layers=20,
        decisions=EMPTY_DECISIONS,
        input_dimensions=(3, 32, 32),
        action_count=0,
    )

    decisions, masked_logits = sample_actions(ctx)

    assert decisions.action_choice == StandardAction.ADD_LAYER


def test_layertype_no_none_when_adding_layer():
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
        logits=get_logits(next_action=StandardAction.ADD_LAYER),
        observation=get_obs(config, max_layers=20),
        slices=get_logit_slices(max_layers=20),
        sampling_strategy=standard_stochastic_sampling,
        max_layers=20,
        decisions=EMPTY_DECISIONS,
        input_dimensions=(3, 32, 32),
        action_count=1,
    )

    decisions, masked_logits = sample_actions(ctx)

    assert decisions.layer_type_choice != LayerType.NONE


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
        slices=get_logit_slices(max_layers=MAX_LAYERS),
        sampling_strategy=standard_stochastic_sampling,
        max_layers=MAX_LAYERS,
        decisions=EMPTY_DECISIONS,
        input_dimensions=(3, 32, 32),
        action_count=MAX_LAYERS,
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
                activation=ActivationFunction.RELU,
            ),
        ]
    )

    ctx = MaskContext(
        logits=get_logits(),
        observation=get_obs(config, max_layers=MAX_LAYERS),
        slices=get_logit_slices(max_layers=MAX_LAYERS),
        sampling_strategy=standard_stochastic_sampling,
        max_layers=MAX_LAYERS,
        decisions=EMPTY_DECISIONS,
        input_dimensions=(3, 32, 32),
        action_count=2,
    )

    decisions, masked_logits = sample_actions(ctx)

    assert decisions.layer_type_choice != LayerType.CONV
    assert decisions.layer_type_choice != LayerType.POOL
    
def test_no_pool_without_conv():
    MAX_LAYERS = 20
    config = NetworkConfig(
        layers=[
            LayerConfig(
                layer_type=LayerType.LINEAR,
                linear_units=LinearUnits.LU_64,
                activation=ActivationFunction.RELU,
            ),
        ]
    )

    ctx = MaskContext(
        logits=get_logits(),
        observation=get_obs(config, max_layers=MAX_LAYERS),
        slices=get_logit_slices(max_layers=MAX_LAYERS),
        sampling_strategy=standard_stochastic_sampling,
        max_layers=MAX_LAYERS,
        decisions=EMPTY_DECISIONS,
        input_dimensions=(3, 32, 32),
        action_count=1,
    )

    decisions, masked_logits = sample_actions(ctx)

    assert decisions.layer_type_choice != LayerType.POOL


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
                activation=ActivationFunction.RELU,
            ),
        ]
    )

    ctx = MaskContext(
        logits=get_logits(next_action=StandardAction.ADD_LAYER),
        observation=get_obs(config, max_layers=MAX_LAYERS),
        slices=get_logit_slices(max_layers=MAX_LAYERS),
        sampling_strategy=standard_stochastic_sampling,
        max_layers=MAX_LAYERS,
        decisions=EMPTY_DECISIONS,
        input_dimensions=(3, 32, 32),
        action_count=2,
    )

    decisions, masked_logits = sample_actions(ctx)

    assert decisions.out_channels_choice == OutChannels.NONE


def test_outputchannelse_notnone_given_add_conv():
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
        logits=get_logits(next_action=StandardAction.ADD_LAYER, next_layer=LayerType.CONV),
        observation=get_obs(config, max_layers=MAX_LAYERS),
        slices=get_logit_slices(max_layers=MAX_LAYERS),
        sampling_strategy=standard_stochastic_sampling,
        max_layers=MAX_LAYERS,
        decisions=EMPTY_DECISIONS,
        input_dimensions=(3, 32, 32),
        action_count=1,
    )

    decisions, masked_logits = sample_actions(ctx)

    assert decisions.out_channels_choice != OutChannels.NONE


def test_outputchannels_notnone_given_add_pool():
    MAX_LAYERS = 20
    config = NetworkConfig(
        layers=[
            LayerConfig(
                layer_type=LayerType.CONV,
                out_channels=OutChannels.CH_16,
                kernel_size=KernelSize.KS_3,
                stride=Stride.S_2,
                activation=ActivationFunction.RELU,
            ),
        ]
    )

    ctx = MaskContext(
        logits=get_logits(next_action=StandardAction.ADD_LAYER, next_layer=LayerType.POOL),
        observation=get_obs(config, max_layers=MAX_LAYERS),
        slices=get_logit_slices(max_layers=MAX_LAYERS),
        sampling_strategy=standard_stochastic_sampling,
        max_layers=MAX_LAYERS,
        decisions=EMPTY_DECISIONS,
        input_dimensions=(3, 32, 32),
        action_count=1,
    )

    decisions, masked_logits = sample_actions(ctx)
    print("Decisions:", decisions)

    assert decisions.out_channels_choice != OutChannels.NONE


def test_kernel_size_notnone_given_add_conv():
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
        logits=get_logits(next_action=StandardAction.ADD_LAYER, next_layer=LayerType.CONV),
        observation=get_obs(config, max_layers=MAX_LAYERS),
        slices=get_logit_slices(max_layers=MAX_LAYERS),
        sampling_strategy=standard_stochastic_sampling,
        max_layers=MAX_LAYERS,
        decisions=EMPTY_DECISIONS,
        input_dimensions=(3, 32, 32),
        action_count=1,
    )

    decisions, masked_logits = sample_actions(ctx)
    print("Decisions:", decisions)

    assert decisions.kernel_size_choice != KernelSize.NONE


def test_kernel_size_notnone_given_add_pool():
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
        logits=get_logits(next_action=StandardAction.ADD_LAYER, next_layer=LayerType.POOL),
        observation=get_obs(config, max_layers=MAX_LAYERS),
        slices=get_logit_slices(max_layers=MAX_LAYERS),
        sampling_strategy=standard_stochastic_sampling,
        max_layers=MAX_LAYERS,
        decisions=EMPTY_DECISIONS,
        input_dimensions=(3, 32, 32),
        action_count=1,
    )

    decisions, masked_logits = sample_actions(ctx)

    assert decisions.kernel_size_choice != KernelSize.NONE


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
                layer_type=LayerType.CONV,
                out_channels=OutChannels.CH_16,
                kernel_size=KernelSize.KS_3,
                stride=Stride.S_2,
                activation=ActivationFunction.RELU,
            ),
        ]
    )

    ctx = MaskContext(
        logits=get_logits(next_action=StandardAction.ADD_LAYER, next_layer=LayerType.CONV),
        observation=get_obs(config, max_layers=MAX_LAYERS),
        slices=get_logit_slices(max_layers=MAX_LAYERS),
        sampling_strategy=standard_stochastic_sampling,
        max_layers=MAX_LAYERS,
        decisions=EMPTY_DECISIONS,
        input_dimensions=(3, 32, 32),
        action_count=4,
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
        slices=get_logit_slices(max_layers=MAX_LAYERS),
        sampling_strategy=standard_stochastic_sampling,
        max_layers=MAX_LAYERS,
        decisions=EMPTY_DECISIONS,
        input_dimensions=(3, 32, 32),
        action_count=4,
    )

    decisions, masked_logits = sample_actions(ctx)

    assert decisions.layer_type_choice == LayerType.CONV
    assert (
        decisions.kernel_size_choice == KernelSize.KS_3
        or decisions.kernel_size_choice == KernelSize.KS_1
    )


def test_linear_units_notnone_given_add_linear():
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
        logits=get_logits(next_action=StandardAction.ADD_LAYER, next_layer=LayerType.LINEAR),
        observation=get_obs(config, max_layers=MAX_LAYERS),
        slices=get_logit_slices(max_layers=MAX_LAYERS),
        sampling_strategy=standard_stochastic_sampling,
        max_layers=MAX_LAYERS,
        decisions=EMPTY_DECISIONS,
        input_dimensions=(3, 32, 32),
        action_count=1,
    )

    decisions, masked_logits = sample_actions(ctx)

    assert decisions.linear_units_choice != LinearUnits.NONE


"""
def test_skip_none_allowed_when_action_count_lte_2():
    # When action_count <= 2, all skip options except 'no skip' should be masked
    empty_config = NetworkConfig(layers=[])

    ctx = MaskContext(
        logits=get_logits(next_action=StandardAction.ADD_LAYER),
        observation=get_obs(empty_config, max_layers=20),
        slices=get_logit_slices(max_layers=20),
        sampling_strategy=standard_stochastic_sampling,
        max_layers=20,
        decisions=EMPTY_DECISIONS,
        input_dimensions=(3, 32, 32),
        action_count=0,
    )

    decisions, masked_logits = sample_actions(ctx)

    # All skip indices except the last should be -inf
    for i in range(0, ctx.max_layers - 1):
        assert np.isneginf(masked_logits[ctx.slices.skip_connection[i]])
    # Last index corresponds to 'no skip' and should be allowed (set to 1)
    assert masked_logits[ctx.slices.skip_connection[ctx.max_layers - 1]] == 1.0


def test_skip_masking_1():
    MAX_LAYERS = 20
    cfg = NetworkConfig(
        layers=[
            LayerConfig(
                layer_type=LayerType.CONV,
                out_channels=OutChannels.CH_16,
                kernel_size=KernelSize.KS_3,
                stride=Stride.S_1,
                activation=ActivationFunction.RELU,
            )
            for _ in range(5)
        ]
        + [
            LayerConfig(
                layer_type=LayerType.CONV,
                out_channels=OutChannels.CH_16,
                kernel_size=KernelSize.KS_3,
                stride=Stride.S_2,  # different stride -> different output dims
                activation=ActivationFunction.RELU,
            )
        ]
        + [
            LayerConfig(
                layer_type=LayerType.CONV,
                out_channels=OutChannels.CH_16,
                kernel_size=KernelSize.KS_3,
                stride=Stride.S_1,
                activation=ActivationFunction.RELU,
            )
            for _ in range(5)
        ]
    )
    layer_count = len(cfg.layers)
    ctx = MaskContext(
        logits=get_logits(
            next_action=StandardAction.ADD_LAYER,
            next_layer=LayerType.CONV,
            next_kernel_size=KernelSize.KS_3,
            next_stride=Stride.S_1,
        ),
        observation=get_obs(cfg, max_layers=MAX_LAYERS),
        slices=get_logit_slices(max_layers=MAX_LAYERS),
        sampling_strategy=standard_stochastic_sampling,
        max_layers=MAX_LAYERS,
        decisions=EMPTY_DECISIONS,
        input_dimensions=(3, 32, 32),
        action_count=layer_count,
    )

    decisions, masked_logits = sample_actions(ctx)

    # The skip option for the first 5 layers should be disabled because dims differ compared to the new layer being added.
    for i in range(5):
        assert np.isneginf(masked_logits[ctx.slices.skip_connection[i]])
    # The skip option for the following 5 layers should be allowed since dims match that of the new layer being added.
    for i in range(5, 10):
        assert not np.isneginf(masked_logits[ctx.slices.skip_connection[i]])

    assert np.isneginf(masked_logits[ctx.slices.skip_connection[layer_count - 1]])

    for i in range(layer_count, MAX_LAYERS - 1):
        assert np.isneginf(masked_logits[ctx.slices.skip_connection[i]])

    assert not np.isneginf(masked_logits[ctx.slices.skip_connection[MAX_LAYERS - 1]])


def test_skip_masking_2():
    cfg = NetworkConfig(
        layers=[
            LayerConfig(
                layer_type=LayerType.CONV,
                out_channels=OutChannels.CH_16,
                kernel_size=KernelSize.KS_3,
                stride=Stride.S_2,
                activation=ActivationFunction.RELU,
            )
            for _ in range(5)
        ]
    )

    MAX_LAYERS = 20
    action_count = len(cfg.layers)

    ctx = MaskContext(
        logits=get_logits(),
        observation=get_obs(cfg, max_layers=MAX_LAYERS),
        slices=get_logit_slices(max_layers=MAX_LAYERS),
        sampling_strategy=standard_stochastic_sampling,
        max_layers=MAX_LAYERS,
        decisions=EMPTY_DECISIONS,
        input_dimensions=(3, 32, 32),
        action_count=action_count,
    )

    decisions, masked_logits = sample_actions(ctx)

    if decisions.action_choice == StandardAction.NONE:
        for i in range(MAX_LAYERS - 1):
            assert np.isneginf(masked_logits[ctx.slices.skip_connection[i]])

    if decisions.layer_type_choice == LayerType.LINEAR:
        for i in range(action_count - 1):
            assert not np.isneginf(masked_logits[ctx.slices.skip_connection[i]])
        for i in range(action_count - 1, MAX_LAYERS - 1):
            assert np.isneginf(masked_logits[ctx.slices.skip_connection[i]])
    else:
        for i in range(MAX_LAYERS - 1):
            assert np.isneginf(masked_logits[ctx.slices.skip_connection[i]])

    assert not np.isneginf(masked_logits[ctx.slices.skip_connection[MAX_LAYERS - 1]])


def test_skip_masking_3():
    cfg = NetworkConfig(
        layers=[
            LayerConfig(
                layer_type=LayerType.CONV,
                out_channels=OutChannels.CH_16,
                kernel_size=KernelSize.KS_3,
                stride=Stride.S_2,
                activation=ActivationFunction.RELU,
            )
            for _ in range(5)
        ]
        + [LayerConfig(layer_type=LayerType.LINEAR, linear_units=LinearUnits.LU_128)]
    )

    MAX_LAYERS = 20
    action_count = len(cfg.layers)

    ctx = MaskContext(
        logits=get_logits(),
        observation=get_obs(cfg, max_layers=MAX_LAYERS),
        slices=get_logit_slices(max_layers=MAX_LAYERS),
        sampling_strategy=standard_stochastic_sampling,
        max_layers=MAX_LAYERS,
        decisions=EMPTY_DECISIONS,
        input_dimensions=(3, 32, 32),
        action_count=action_count,
    )

    decisions, masked_logits = sample_actions(ctx)

    if decisions.action_choice == StandardAction.ADD_LAYER:
        for i in range(action_count - 1):
            assert not np.isneginf(masked_logits[ctx.slices.skip_connection[i]])
        for i in range(action_count - 1, MAX_LAYERS - 1):
            assert np.isneginf(masked_logits[ctx.slices.skip_connection[i]])
    else:
        for i in range(MAX_LAYERS):
            assert np.isneginf(masked_logits[ctx.slices.skip_connection[i]])
"""
