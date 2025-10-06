import numpy as np

from model_module.action_builder_utils import (
    ActionStrategy,
    add_layer_complete,
    custom_sample_example,
    get_logit_slices,
)


class ActionBuilder:
    def __init__(self, max_layers: int, strategy: ActionStrategy):
        self.slices = get_logit_slices(max_layers)  # Fixed function name
        self.strategy = strategy
        self.max_layers = max_layers

    def build_action(self, action_output: np.ndarray) -> list[int]:
        if self.strategy == ActionStrategy.ONLY_ADD_SEQUENTIAL:
            return add_layer_complete(action_output, self.slices, custom_sample_example)
        elif self.strategy == ActionStrategy.ADD_REMOVE_MODIFY:
            raise NotImplementedError("ADD_REMOVE_MODIFY strategy is not implemented yet.")
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
