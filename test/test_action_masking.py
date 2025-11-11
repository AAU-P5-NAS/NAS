import numpy as np
from src.action_masking.action_masking import sample_actions
from src.action_masking.action_masking_utils import (
    MaskContext,
    get_logit_slices,
    standard_stochastic_sampling,
)
from src.utils.cnn_builder import flatten_cnn_config
from src.utils.network_utils import (
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


def get_logits():
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
    return np.random.randn(sum).astype(np.float32)


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
    pass
