import numpy as np

from utils.action_builder_utils import (
    ActionStrategy,
    MaskContext,
    build_action_add_layer_sequential,
    get_logit_slices,
)


class ActionBuilder:
    def __init__(self, max_layers: int, strategy: ActionStrategy):
        self.slices = get_logit_slices(max_layers)  # Fixed function name
        self.strategy = strategy
        self.max_layers = max_layers

    def build_action(self, action_output: np.ndarray, observation: list[int]):
        ctx = MaskContext(
            logits=action_output,
            observation=observation,
            slices=self.slices,
            action_strategy=self.strategy,
            sampling_strategy=np.argmax,
            max_layers=self.max_layers,
        )

        match self.strategy:
            case ActionStrategy.ADD_LAYER_SEQUENTIAL:
                return build_action_add_layer_sequential(ctx)
            case ActionStrategy.ADD_REMOVE_MODIFY:
                raise NotImplementedError("ADD_REMOVE_MODIFY strategy is not implemented yet.")
            case _:
                raise ValueError(f"Unknown strategy: {self.strategy}")
