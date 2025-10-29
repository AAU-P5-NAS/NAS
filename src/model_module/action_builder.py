import numpy as np

from src.utils.action_builder_utils import (
    MaskContext,
    build_action_add_layer_sequential,
    get_logit_slices,
)
from src.utils.network_utils import EMPTY_DECISIONS


def transform_logits_to_action(action_output: np.ndarray, observation: np.ndarray, max_layers: int):
    ctx = MaskContext(
        logits=action_output,
        observation=observation,
        slices=get_logit_slices(max_layers),
        sampling_strategy=np.argmax,
        max_layers=max_layers,
        decisions=EMPTY_DECISIONS,
    )
    return build_action_add_layer_sequential(ctx)
