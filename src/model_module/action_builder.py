from typing import Optional
import numpy as np

from src.utils.action_builder_utils import (
    MaskContext,
    build_action_add_layer_sequential,
    get_logit_slices,
)
from src.utils.network_utils import EMPTY_DECISIONS, Decisions


def standard_stochastic_sampling(logits: np.ndarray) -> int:
    """Samples an index from the logits using a softmax distribution."""
    exp_logits = np.exp(logits - np.max(logits))  # ensure positive numbers
    probs = exp_logits / exp_logits.sum()  # compute probabilities
    return np.random.choice(len(logits), p=probs)  # sample index based on probs and return idx


def transform_logits_to_action(
    action_output: np.ndarray,
    observation: np.ndarray,
    max_layers: int,
    dimensions: tuple[int, int, int],
    actions_taken: int,
) -> Optional[Decisions]:
    ctx = MaskContext(
        logits=action_output,
        observation=observation,
        slices=get_logit_slices(max_layers),
        sampling_strategy=standard_stochastic_sampling,
        max_layers=max_layers,
        decisions=EMPTY_DECISIONS,
        input_dimensions=dimensions,
        action_count=actions_taken,
    )
    return build_action_add_layer_sequential(ctx)
