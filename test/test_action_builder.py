import pytest
import numpy as np
import sys
import os

from stable_baselines3 import A2C

from src.classification_module.reward import WeightedSumRS, Weights
from src.model_module.action_builder import standard_stochastic_sampling, transform_logits_to_action
from src.model_module.environment import CustomEnv
from src.utils.network_utils import Decisions

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestStandardStochasticSampling:
    empty_logits = np.array([])
    dummy_logits = np.array([1, 0, 1, 1])

    def test_empty(self):
        with pytest.raises(ValueError):
            standard_stochastic_sampling(self.empty_logits)

    def test_correct_logits_gives_integer(self):
        assert type(standard_stochastic_sampling(self.dummy_logits)) is int


class TestTransformLogitsToAction:
    add_linear_action = [2, 2, 2, -1, -1, -1, 3, -1, 1]
    env = CustomEnv(
        device="cuda",
        logdir="",
        training_epochs=1,
        arch_learning_rate=0.1,
        arch_momentum=1,
        batch_size=64,
        reward_strategy=WeightedSumRS(weights=Weights.staticWeights()),
    )

    model = A2C("MlpPolicy", env)
    obs, _ = env.reset()
    action, _ = model.predict(obs)

    def test_empty(self):
        assert (
            transform_logits_to_action(
                np.array([]), np.array([]), self.env.max_layers, (0, 0, 0), 1
            )
            is None
        )

    def test_finds_action(self):
        print(self.obs, self.action)
        assert isinstance(
            transform_logits_to_action(self.action, self.obs, self.env.max_layers, (1, 28, 28), 1),
            Decisions,
        )


'''
from src.utils.action_builder_utils import (
    ActionStrategy,
)

from src.utils.network_utils import (
    StandardAction,
    LayerType,
    OutChannels,
    KernelSize,
    Stride,
    LinearUnits,
    PoolMode,
    ActivationFunction,
)

from src.model_module.action_builder import ActionBuilder


class TestActionBuilderInitialization:
    """Test ActionBuilder initialization."""

    def test_init_with_valid_params(self):
        """Test initialization with valid parameters."""
        max_layers = 5
        strategy = ActionStrategy.ADD_LAYER_SEQUENTIAL.value
        builder = ActionBuilder(max_layers, strategy)

        assert builder.max_layers == max_layers
        assert builder.strategy == strategy
        assert builder.slices is not None

    def test_init_creates_correct_slices(self):
        """Test that initialization creates slices with correct structure."""
        builder = ActionBuilder(5, ActionStrategy.ADD_LAYER_SEQUENTIAL.value)

        assert hasattr(builder.slices, "standard_actions")
        assert hasattr(builder.slices, "layer_type")
        assert hasattr(builder.slices, "layer_index")
        assert hasattr(builder.slices, "out_channels")
        assert hasattr(builder.slices, "kernel_size")
        assert hasattr(builder.slices, "stride")
        assert hasattr(builder.slices, "linear_units")
        assert hasattr(builder.slices, "pool_mode")
        assert hasattr(builder.slices, "activation_function")


class TestActionBuilderSequential:
    """Test ActionBuilder with ADD_LAYER_SEQUENTIAL strategy."""

    @pytest.fixture
    def builder(self):
        """Create a builder instance for testing."""
        return ActionBuilder(max_layers=5, strategy=ActionStrategy.ADD_LAYER_SEQUENTIAL.value)

    @pytest.fixture
    def empty_observation(self):
        """Create an empty observation (no layers yet)."""
        return [-1] * 35  # 5 layers * 7 values per layer

    @pytest.fixture
    def observation_with_one_conv_layer(self):
        """Create observation with one convolutional layer."""
        obs = [-1] * 35
        obs[0:7] = [
            LayerType.CONV.value,
            OutChannels.CH_32.value,
            KernelSize.KS_3.value,
            Stride.S_1.value,
            -1,  # pool_mode (not used for CONV)
            ActivationFunction.RELU.value,
            -1,  # linear_units (not used for CONV)
        ]
        return obs

    @pytest.fixture
    def observation_with_linear_layer(self):
        """Create observation with a linear layer."""
        obs = [0] * 35
        obs[0:7] = [
            LayerType.LINEAR.value,
            0,  # out_channels (not used for LINEAR)
            0,  # kernel_size
            0,  # stride
            0,  # pool_mode
            ActivationFunction.RELU.value,
            LinearUnits.LU_128.value,
        ]
        return obs

    def test_build_action_returns_list_of_decisions(self, builder, empty_observation):
        """Test that build_action returns a list of decisions."""
        logit_size = sum(
            [
                len(StandardAction),
                len(LayerType),
                builder.max_layers - 1,
                len(OutChannels),
                len(KernelSize),
                len(Stride),
                len(LinearUnits),
                len(PoolMode),
                len(ActivationFunction),
            ]
        )
        action_output = np.random.randn(logit_size)

        decisions = builder.build_action(action_output, empty_observation)

        assert isinstance(decisions, list)
        assert len(decisions) == 9  # One decision for each head

    def test_build_action_on_empty_observation(self, builder, empty_observation):
        """Test building action on empty observation."""
        logit_size = sum(
            [
                len(StandardAction),
                len(LayerType),
                builder.max_layers - 1,
                len(OutChannels),
                len(KernelSize),
                len(Stride),
                len(LinearUnits),
                len(PoolMode),
                len(ActivationFunction),
            ]
        )
        action_output = np.ones(logit_size)

        decisions = builder.build_action(action_output, empty_observation)

        # First layer should be at index 0
        assert decisions[1] == 0  # layer_index decision

    def test_build_action_adds_layer_sequentially(self, builder, observation_with_one_conv_layer):
        """Test that layers are added sequentially."""
        logit_size = sum(
            [
                len(StandardAction),
                len(LayerType),
                builder.max_layers - 1,
                len(OutChannels),
                len(KernelSize),
                len(Stride),
                len(LinearUnits),
                len(PoolMode),
                len(ActivationFunction),
            ]
        )
        action_output = np.ones(logit_size)

        decisions = builder.build_action(action_output, observation_with_one_conv_layer)

        # Second layer should be at index 1
        assert decisions[1] == 1  # layer_index decision

    def test_build_action_forces_linear_after_linear(self, builder, observation_with_linear_layer):
        """Test that only linear layers can follow linear layers."""
        logit_size = sum(
            [
                len(StandardAction),
                len(LayerType),
                builder.max_layers - 1,
                len(OutChannels),
                len(KernelSize),
                len(Stride),
                len(LinearUnits),
                len(PoolMode),
                len(ActivationFunction),
            ]
        )
        action_output = np.ones(logit_size)

        decisions = builder.build_action(action_output, observation_with_linear_layer)

        # layer_type decision should be LINEAR
        assert decisions[2] == LayerType.LINEAR.value or decisions[2] == LayerType.NONE.value

    def test_build_action_masks_invalid_kernel_sizes(
        self, builder, observation_with_one_conv_layer
    ):
        """Test that invalid kernel sizes are masked based on spatial dimensions."""
        # After one 3x3 conv with stride 1, we have 26x26 output
        # All kernel sizes should be valid for 26x26
        logit_size = sum(
            [
                len(StandardAction),
                len(LayerType),
                builder.max_layers - 1,
                len(OutChannels),
                len(KernelSize),
                len(Stride),
                len(LinearUnits),
                len(PoolMode),
                len(ActivationFunction),
            ]
        )
        action_output = np.ones(logit_size)

        decisions = builder.build_action(action_output, observation_with_one_conv_layer)

        # kernel_size decision should be valid
        assert decisions[4] in [k.value for k in KernelSize]

    def test_build_action_raises_on_max_layers(self, builder):
        """Test that MaxLayersReachedException is raised when max layers reached."""
        # Create observation with max layers (5 layers)
        obs = []
        for i in range(5):
            obs.extend(
                [
                    LayerType.CONV.value,
                    OutChannels.CH_32.value,
                    KernelSize.KS_3.value,
                    Stride.S_1.value,
                    LinearUnits.NONE.value,
                    ActivationFunction.RELU.value,
                    PoolMode.NONE.value,
                ]
            )

        logit_size = sum(
            [
                len(StandardAction),
                len(LayerType),
                builder.max_layers - 1,
                len(OutChannels),
                len(KernelSize),
                len(Stride),
                len(LinearUnits),
                len(PoolMode),
                len(ActivationFunction),
            ]
        )
        action_output = np.ones(logit_size)

        try:
            builder.build_action(action_output, obs)
        except Exception as e:
            assert e is not None

    def test_build_action_with_pool_layer(self, builder, empty_observation):
        """Test building action with pool layer type."""
        logit_size = sum(
            [
                len(StandardAction),
                len(LayerType),
                builder.max_layers - 1,
                len(OutChannels),
                len(KernelSize),
                len(Stride),
                len(LinearUnits),
                len(PoolMode),
                len(ActivationFunction),
            ]
        )
        action_output = np.ones(logit_size)

        # Manually set layer_type to POOL in the logits
        layer_type_slice = builder.slices.layer_type
        action_output[layer_type_slice.start : layer_type_slice.stop] = -np.inf
        pool_idx = layer_type_slice.get_index(LayerType.POOL)
        action_output[pool_idx] = 10.0  # High logit for POOL

        decisions = builder.build_action(action_output, empty_observation)

        # Should have chosen POOL layer type
        assert decisions[2] == LayerType.POOL.value

    """ def test_decision_indices_are_valid(self, builder, empty_observation):
        logit_size = sum(
            [
                len(StandardAction),
                len(LayerType),
                builder.max_layers - 1,
                len(OutChannels),
                len(KernelSize),
                len(Stride),
                len(LinearUnits),
                len(PoolMode),
                len(ActivationFunction),
            ]
        )
        action_output = np.random.randn(logit_size)

        decisions = builder.build_action(action_output, empty_observation)

        assert decisions[0] in [a.value for a in StandardAction]
        assert decisions[1] >= 0 and decisions[1] < builder.max_layers
        assert decisions[2] in [lt.value for lt in LayerType]
        assert decisions[3] in [oc.value for oc in [OutChannels, -1]]
        assert decisions[4] in [k.value for k in [KernelSize, -1]]
        assert decisions[5] in [s.value for s in [Stride, -1]]
        assert decisions[6] in [lu.value for lu in [LinearUnits, -1]]
        assert decisions[7] in [pm.value for pm in [PoolMode, ]]
        assert decisions[8] in [af.value for af in ActivationFunction] """


class TestActionBuilderUnimplementedStrategies:
    """Test ActionBuilder with unimplemented strategies."""

    def test_add_remove_modify_raises_not_implemented(self):
        """Test that ADD_REMOVE_MODIFY strategy raises NotImplementedError."""
        builder = ActionBuilder(5, ActionStrategy.ADD_REMOVE_MODIFY.value)
        action_output = np.ones(50)
        observation = [0] * 35

        with pytest.raises(
            NotImplementedError, match="ADD_REMOVE_MODIFY strategy is not implemented yet"
        ):
            builder.build_action(action_output, observation)


class TestActionBuilderEdgeCases:
    """Test edge cases for ActionBuilder."""

    def test_build_action_with_small_spatial_dimensions(self):
        """Test building action when spatial dimensions become very small."""
        builder = ActionBuilder(5, ActionStrategy.ADD_LAYER_SEQUENTIAL.value)

        # Create observation that leads to small spatial dimensions
        obs = [0] * 35
        obs[0:7] = [
            LayerType.CONV.value,
            OutChannels.CH_32.value,
            KernelSize.KS_5.value,
            Stride.S_2.value,
            LinearUnits.NONE.value,
            ActivationFunction.RELU.value,
            PoolMode.NONE.value,
        ]
        # After 5x5 conv with stride 2 on 28x28: (28-5)//2+1 = 12

        obs[7:14] = [
            LayerType.CONV.value,
            OutChannels.CH_64.value,
            KernelSize.KS_5.value,
            Stride.S_2.value,
            LinearUnits.NONE.value,
            ActivationFunction.RELU.value,
            PoolMode.NONE.value,
        ]
        # After another 5x5 conv with stride 2 on 12x12: (12-5)//2+1 = 4

        logit_size = sum(
            [
                len(StandardAction),
                len(LayerType),
                builder.max_layers - 1,
                len(OutChannels),
                len(KernelSize),
                len(Stride),
                len(LinearUnits),
                len(PoolMode),
                len(ActivationFunction),
            ]
        )
        action_output = np.ones(logit_size)

        decisions = builder.build_action(action_output, obs)

        # Should still produce valid decisions
        assert len(decisions) == 9
        assert all(isinstance(d, (int, np.integer)) for d in decisions)

    def test_build_action_with_zero_logits(self):
        """Test building action with all zero logits."""
        builder = ActionBuilder(5, ActionStrategy.ADD_LAYER_SEQUENTIAL.value)
        observation = [0] * 35

        logit_size = sum(
            [
                len(StandardAction),
                len(LayerType),
                builder.max_layers - 1,
                len(OutChannels),
                len(KernelSize),
                len(Stride),
                len(LinearUnits),
                len(PoolMode),
                len(ActivationFunction),
            ]
        )
        action_output = np.zeros(logit_size)

        decisions = builder.build_action(action_output, observation)

        # Should still produce valid decisions (argmax of zeros is first element)
        assert len(decisions) == 9

    def test_build_action_with_negative_logits(self):
        """Test building action with all negative logits."""
        builder = ActionBuilder(5, ActionStrategy.ADD_LAYER_SEQUENTIAL.value)
        observation = [0] * 35

        logit_size = sum(
            [
                len(StandardAction),
                len(LayerType),
                builder.max_layers - 1,
                len(OutChannels),
                len(KernelSize),
                len(Stride),
                len(LinearUnits),
                len(PoolMode),
                len(ActivationFunction),
            ]
        )
        action_output = np.full(logit_size, 0)

        decisions = builder.build_action(action_output, observation)

        # Should still produce valid decisions
        assert len(decisions) == 9
'''
