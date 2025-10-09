import pytest
from src.classification_module.reward import RewardCalculator, Weights
from src.classification_module.metrics import Metrics


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
    calc = RewardCalculator(weights=weights)
    reward = calc.compute_reward(default_metrics)

    # Reward must be between 0 and 1 since all metrics and weights are normalized
    assert 0.0 <= reward <= 1.0


def test_reward_ignores_none_values(default_metrics: Metrics):
    """Test that None metrics are ignored without breaking."""
    metrics = default_metrics
    metrics.precision = None
    calc = RewardCalculator()
    reward = calc.compute_reward(metrics)
    assert isinstance(reward, float)
    assert reward is not None and reward == reward  # Check for NaN


def test_inverse_metrics_scaled(default_metrics: Metrics):
    """Test that flops, runtime, and architecture_size are scaled down."""
    calc = RewardCalculator()
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
    calc = RewardCalculator(weights=zero_weights)
    reward = calc.compute_reward(default_metrics)
    assert reward == 0.0


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
    calc = RewardCalculator(weights=custom_weights)
    reward = calc.compute_reward(default_metrics)
    assert 0.0 <= reward <= 1.0
    # Reward should depend primarily on accuracy and test_loss
    assert reward > 0
