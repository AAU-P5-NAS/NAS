import gymnasium as gym
import numpy as np
from gymnasium import spaces
from enum import Enum


class Operation(Enum):
    NO_OP = 0
    ADD_LAYER = 1
    REMOVE_LAYER = 2
    ADD_NEURON = 3
    REMOVE_NEURON = 4
    CHANGE_ACTIVATION_FUNCTION = 5
    SET_EXACT_NEURONS = 6


class CustomEnv(gym.Env):
    """A custom environment for reinforcement learning."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        render_mode="console",
        max_layers=10,
        max_neurons=1024,
        n_activations=5,
        neuron_bucket_count=6,
    ):
        super().__init__()
        self.render_mode = render_mode

        self.max_layers = max_layers
        self.max_neurons = max_neurons
        self.n_activations = n_activations
        self.neuron_bucket_count = neuron_bucket_count

        # Feature vector size: is_present, layer_type, neurons_norm, activation_idx_norm, dropout, params_share, position_norm
        self.features_per_layer = 7
        self.observation_space = spaces.Dict(
            {
                "layers": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.max_layers, self.features_per_layer),
                    dtype=np.float32,
                ),
                "global": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
            }
        )

        # Action Space
        N_OPS = Operation.__len__()
        PARAM_CHOICES = max(self.n_activations, self.neuron_bucket_count)
        self.action_space = spaces.MultiDiscrete([N_OPS, self.max_layers, PARAM_CHOICES])

        self.hidden_layers: list[dict[str, int]]
        self.prev_val_acc = 0.0
        self.max_total_params = self.max_layers * self.max_neurons

    def reset(self, seed=None, options=None, initial_fnn=None):
        super().reset(seed=seed)
        if initial_fnn is None:
            self.hidden_layers = [{"neurons": 64, "activation": 1, "dropout": 0}]
        else:
            self.hidden_layers = self._encode_from_external(initial_fnn)

    def step(self, action: list[int]):
        op, layer_index, param = Operation(action[0]), int(action[1]), int(action[2])
        is_illegal = not self._apply_action(op, layer_index, param)

        # evaluation
        val_acc = self._evaluate_proxy()
        reward = (val_acc - self.prev_val_acc) - 0.01 * (
            self._total_params() / self.max_total_params
        )

        if is_illegal:
            reward -= 0.02

        self.prev_val_acc = val_acc

        done = False  # termination logic
        info = {"val_acc": float(val_acc), "params": int(self._total_params())}
        return self._get_obs(), float(reward), done, info

    def _get_obs(self) -> dict:
        layers = np.zeros((self.max_layers, self.features_per_layer), dtype=np.float32)
        total_params = self._total_params()
        for i in range(self.max_layers):
            if i < len(self.hidden_layers):
                hiden_layer = self.hidden_layers[i]
                layers[i, 0] = 1.0  # is_present
                layers[i, 1] = 1.0  # layer_type
                layers[i, 2] = min(hiden_layer["neurons"] / self.max_neurons, 1.0)
                layers[i, 3] = hiden_layer["activation"] / max(1, self.n_activations)
                layers[i, 4] = float(hiden_layer["dropout"])
                layers[i, 5] = (
                    (hiden_layer["neurons"] / max(1, total_params)) if total_params > 0 else 0.0
                )
                layers[i, 6] = i / max(1, self.max_layers - 1)
            else:
                pass  # remains zeros with is_present=0

        global_vec = np.array(
            [
                min(total_params / self.max_total_params, 1.0),
                self.prev_val_acc,
                0.0,  # steps_since_eval_norm (if tracked)
                1.0,  # remaining budget (placeholder)
            ],
            dtype=np.float32,
        )

        return {"layers": layers, "global": global_vec}

    def _apply_action(self, op: Operation, layer_index: int, param: int):
        """returns True if legal & applied; False if illegal (no change)"""

        is_layer_index_legal = 0 <= layer_index < len(self.hidden_layers)

        is_legal: bool
        match op:
            case Operation.NO_OP:
                is_legal = True

            case Operation.ADD_LAYER:
                is_legal = len(self.hidden_layers) < self.max_layers
                if is_legal:
                    neurons = self._bucket_to_neurons(param)
                    new_layer = {"neurons": neurons, "activation": 0, "dropout": 0}

                    insert_pos = min(layer_index, len(self.hidden_layers))
                    self.hidden_layers.insert(insert_pos, new_layer)

            case Operation.REMOVE_LAYER:
                is_legal = is_layer_index_legal
                if is_legal:
                    self.hidden_layers.pop(layer_index)

            case Operation.ADD_NEURON:
                if not is_layer_index_legal:
                    is_legal = False
                else:
                    is_legal = self.hidden_layers[layer_index]["neurons"] == self.max_neurons

                    if is_legal:
                        
