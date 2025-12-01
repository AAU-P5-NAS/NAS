import pytest
import torch
from src.environment.reward.tchebycheff import TchebycheffRS
from src.environment.reward.weighted_sum import WeightedSumRS
from src.environment.reward.reward import Weights
from src.environment.metrics import Evaluator, Metrics
from src.utils.layer_config import (
    ActivationFunction,
    KernelSize,
    LayerConfig,
    LayerType,
    LinearUnits,
    OutChannels,
    Stride,
)
from src.utils.architecture import Architecture
from src.environment.reward.archive_pareto_dom import DominanceNoveltyRS
from src.utils.network_config import NetworkConfig


@pytest.fixture
def evaluator():
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(10, 3, 28, 28), torch.randint(0, 10, (10,)))
    )
    return Evaluator(
        num_classes=10,
        loss_function=torch.nn.CrossEntropyLoss(),
        dataloaders=(dl, dl),
        dimensions=(3, 28, 28),
        device=torch.device("cpu"),
    )


@pytest.fixture
def default_metrics() -> Metrics:
    """Fixture returning a dummy metrics object with known values."""
    return Metrics(
        accuracy=0.8,
        precision=0.75,
        recall=0.7,
        f1_score=0.78,
        test_loss=0.3,
        flops=100,
        runtime=2.0,
        architecture_size=50,
    )


def test_reward_calculator_normalization(default_metrics: Metrics):
    """Test that weights are normalized to sum to 1."""
    weights = Weights(
        accuracy=6.0,
        f1_score=10.0,
        test_loss=5.0,
        flops=2.0,
        runtime=3.0,
        architecture_size=1.0,
    )
    calc = WeightedSumRS(weights=weights)
    reward = calc.compute_reward(default_metrics)

    # Reward must be between 0 and 1 since all metrics and weights are normalized
    assert 0.0 <= reward <= 1.0


def test_reward_ignores_none_values(default_metrics: Metrics):
    """Test that None metrics are ignored without breaking."""
    metrics = default_metrics
    metrics.precision = None
    w = Weights.staticWeights()
    calc = WeightedSumRS(w)
    reward = calc.compute_reward(metrics)
    assert isinstance(reward, float)
    assert reward is not None and reward == reward  # Check for NaN

    w = Weights.dynamicWeightsSampler()
    calc = TchebycheffRS(w)
    reward = calc.compute_reward(metrics)
    assert isinstance(reward, float)
    assert reward is not None and reward == reward  # Check for NaN


def test_inverse_metrics_scaled(default_metrics: Metrics):
    """Test that flops, runtime, and architecture_size are scaled down."""

    w = Weights.staticWeights()
    calc = WeightedSumRS(w)
    reward = calc.compute_reward(default_metrics)
    # If those values are large, the reward should be reduced
    smaller_flops = default_metrics.model_copy(update={"flops": 1.0})
    higher_reward = calc.compute_reward(smaller_flops)
    assert higher_reward > reward


def test_zero_weights(default_metrics: Metrics):
    """If all weights are zero, reward should be zero."""
    zero_weights = Weights(
        accuracy=0.0,
        precision=0.0,
        recall=0.0,
        f1_score=0.0,
        test_loss=0.0,
        flops=0.0,
        runtime=0.0,
        architecture_size=0.0,
    )
    calc = WeightedSumRS(weights=zero_weights)
    with pytest.raises(ValueError):
        calc.compute_reward(default_metrics)


def test_partial_metrics(default_metrics: Metrics):
    """Test when only a subset of metrics are used (e.g., accuracy and loss)."""
    custom_weights = Weights(
        accuracy=1.0,
        precision=0.0,
        recall=0.0,
        f1_score=0.0,
        test_loss=1.0,
        flops=0.0,
        runtime=0.0,
        architecture_size=0.0,
    )
    calc = WeightedSumRS(weights=custom_weights)
    reward = calc.compute_reward(default_metrics)
    assert 0.0 <= reward <= 1.0
    # Reward should depend primarily on accuracy and test_loss
    assert reward > 0


def test_negative_weights_raises(default_metrics: Metrics):
    """Test that negative weights raise a ValueError."""
    with pytest.raises(ValueError, match="Weights must be non-negative"):
        Weights(
            accuracy=-1.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            test_loss=0.0,
            flops=0.0,
            runtime=0.0,
            architecture_size=0.0,
        )


def test_static_weights():
    weights = Weights.staticWeights()

    for w in weights.model_dump().values():
        assert isinstance(w, (float, int)) and w is not None


def test_dynamic_weights_randomness():
    weights1 = Weights.dynamicWeightsSampler().model_dump()
    weights2 = Weights.dynamicWeightsSampler().model_dump()
    differences = [weights1[k] != weights2[k] for k in weights1]
    assert any(differences), "Dynamic sampler produced identical weights"


def test_TchebycheffReward_compute(default_metrics: Metrics):
    """Test that weights are normalized to sum to 1."""
    w = Weights.dynamicWeightsSampler()
    calc = TchebycheffRS(w)
    reward = calc.compute_reward(default_metrics)

    assert 0.0 <= reward <= 1.0


def test_dominance_novelty_reward(evaluator):
    reward_strategy = DominanceNoveltyRS(
        weights=Weights.staticWeights(), domnov_weights=Weights.domnovWeights()
    )

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

    architecture = Architecture(
        net_config=config,
        num_classes=10,
        input_dimensions=(3, 32, 32),
    )

    proxy_metrics = evaluator.evaluate_by_proxy(architecture)

    print("Proxy Metrics:", proxy_metrics)

    reward = reward_strategy.compute_reward(proxy_metrics, config)
    print("Dominance-Novelty Reward:", reward)

    assert isinstance(reward, float)
    assert 0.0 <= reward <= 1.0
    assert reward == 0.7  # novelty should be 0, dominance 1, weights 0.3 and 0.7


def test_dominance_novelty_reward_large_network(evaluator):
    reward_strategy = DominanceNoveltyRS(
        weights=Weights.staticWeights(), domnov_weights=Weights.domnovWeights()
    )

    # Create a large network configuration
    config = NetworkConfig(
        layers=[
            LayerConfig(
                layer_type=LayerType.CONV,
                out_channels=OutChannels.CH_64,
                kernel_size=KernelSize.KS_3,
                stride=Stride.S_1,
                activation=ActivationFunction.RELU,
            ),
            LayerConfig(
                layer_type=LayerType.CONV,
                out_channels=OutChannels.CH_128,
                kernel_size=KernelSize.KS_3,
                stride=Stride.S_1,
                activation=ActivationFunction.RELU,
            ),
            LayerConfig(
                layer_type=LayerType.CONV,
                out_channels=OutChannels.CH_256,
                kernel_size=KernelSize.KS_3,
                stride=Stride.S_1,
                activation=ActivationFunction.RELU,
            ),
            LayerConfig(
                layer_type=LayerType.LINEAR,
                linear_units=LinearUnits.LU_512,
                activation=ActivationFunction.RELU,
            ),
            LayerConfig(
                layer_type=LayerType.LINEAR,
                linear_units=LinearUnits.LU_256,
                activation=ActivationFunction.RELU,
            ),
            LayerConfig(
                layer_type=LayerType.LINEAR,
                linear_units=LinearUnits.LU_128,
                activation=ActivationFunction.RELU,
            ),
        ]
    )

    architecture = Architecture(
        net_config=config,
        num_classes=10,
        input_dimensions=(3, 32, 32),
    )

    proxy_metrics = evaluator.evaluate_by_proxy(architecture)

    print("Proxy Metrics for Large Network:", proxy_metrics)

    reward = reward_strategy.compute_reward(proxy_metrics, config)
    print("Dominance-Novelty Reward for Large Network:", reward)

    assert isinstance(reward, float)
    assert 0.0 <= reward <= 1.0


def test_dominance_novelty_reward_multiple_networks(evaluator):
    reward_strategy = DominanceNoveltyRS(
        weights=Weights.staticWeights(), domnov_weights=Weights.domnovWeights()
    )

    # Define multiple network configurations
    configs = [
        NetworkConfig(
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
        ),
        NetworkConfig(
            layers=[
                LayerConfig(
                    layer_type=LayerType.CONV,
                    out_channels=OutChannels.CH_32,
                    kernel_size=KernelSize.KS_3,
                    stride=Stride.S_1,
                    activation=ActivationFunction.RELU,
                ),
                LayerConfig(
                    layer_type=LayerType.LINEAR,
                    linear_units=LinearUnits.LU_128,
                    activation=ActivationFunction.RELU,
                ),
            ]
        ),
        NetworkConfig(
            layers=[
                LayerConfig(
                    layer_type=LayerType.CONV,
                    out_channels=OutChannels.CH_64,
                    kernel_size=KernelSize.KS_3,
                    stride=Stride.S_1,
                    activation=ActivationFunction.RELU,
                ),
                LayerConfig(
                    layer_type=LayerType.LINEAR,
                    linear_units=LinearUnits.LU_256,
                    activation=ActivationFunction.RELU,
                ),
            ]
        ),
    ]

    for i, config in enumerate(configs):
        architecture = Architecture(
            net_config=config,
            num_classes=10,
            input_dimensions=(3, 32, 32),
        )

        proxy_metrics = evaluator.evaluate_by_proxy(architecture)

        print(f"Proxy Metrics for Network {i + 1}:", proxy_metrics)

        reward = reward_strategy.compute_reward(proxy_metrics, config)
        print(f"Dominance-Novelty Reward for Network {i + 1}:", reward)

        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0

    assert 2 == 1
