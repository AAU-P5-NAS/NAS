import numpy as np

""" 
from src.utils.action_builder_utils import (
    MaskContext,
    get_logit_slices,
)
from src.utils.network_utils import EMPTY_DECISIONS

 """


def standard_stochastic_sampling(logits: np.ndarray) -> int:
    """Samples an index from the logits using a softmax distribution."""
    exp_logits = np.exp(logits - np.max(logits))  # ensure positive numbers
    probs = exp_logits / exp_logits.sum()  # compute probabilities
    return np.random.choice(len(logits), p=probs)  # sample index based on probs and return idx


def max_sampling(logits: np.ndarray) -> int:
    """Selects the index with the highest logit value."""
    return int(np.argmax(logits))


""" def transform_logits_to_action(
    action_output: np.ndarray,
    observation: np.ndarray,
    max_layers: int,
    dimensions: tuple[int, int, int],
):
    ctx = MaskContext(
        logits=action_output,
        observation=observation,
        slices=get_logit_slices(),
        sampling_strategy=max_sampling,
        max_layers=max_layers,
        decisions=EMPTY_DECISIONS,
        input_dimensions=dimensions,
    )
    return build_action_add_layer_sequential(ctx) """
