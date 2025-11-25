import pytest
import torch
from torch import nn
from src.environment.metrics import Evaluator, Metrics, InvalidMetricError

NUM_CLASSES = 28

@pytest.fixture
def evaluator():
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(10, 3, 28, 28), torch.randint(0, NUM_CLASSES, (10,))))
    return Evaluator(num_classes=NUM_CLASSES, loss_function=torch.nn.CrossEntropyLoss(), dataloaders=(dl, dl), dimensions=(3, 28, 28), device=torch.device("cpu"))


@pytest.fixture
def dummy_CNN_small():
    return nn.Sequential(nn.ReLU(), nn.Flatten(), nn.Linear(3 * 28 * 28, NUM_CLASSES))


@pytest.fixture
def dummy_CNN_large():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, 1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, 1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(24 * 24 * 32, NUM_CLASSES),
    )


@pytest.fixture
def dummy_data():
    preds = torch.tensor([0, 2, 2, 1])
    targets = torch.tensor([0, 1, 2, 1])
    return preds, targets


def test_compute_flops(evaluator, dummy_CNN_small, dummy_CNN_large):
    small_flops = evaluator.compute_flops(dummy_CNN_small)
    large_flops = evaluator.compute_flops(dummy_CNN_large)

    assert isinstance(small_flops, int)
    assert small_flops > 0
    assert isinstance(large_flops, int)
    assert large_flops > 0
    assert small_flops < large_flops


def test_compute_runtime(evaluator, dummy_CNN_small, dummy_CNN_large):
    small_runtime = evaluator.compute_runtime(dummy_CNN_small, iterations=20)
    large_runtime = evaluator.compute_runtime(dummy_CNN_large, iterations=20)

    assert isinstance(small_runtime, float)
    assert small_runtime > 0
    assert isinstance(large_runtime, float)
    assert large_runtime > 0
    assert small_runtime < large_runtime


@pytest.mark.parametrize("invalid_metric", ["invalid", "speed", "loss_accuracy"])
def test_invalid_metric(evaluator, invalid_metric, dummy_data):
    with pytest.raises(InvalidMetricError):
        preds, targets = dummy_data
        evaluator.calculate_metrics(nn.Identity(), preds, targets, invalid_metric)


def test_calculate_metrics(evaluator, dummy_data):
    preds, targets = dummy_data
    metrics = evaluator.calculate_metrics(
        nn.Identity(), preds, targets, ["accuracy", "precision", "recall", "f1_score"]
    )

    assert isinstance(metrics, Metrics)
    for name in ["accuracy", "precision", "recall", "f1_score"]:
        value = getattr(metrics, name)
        assert isinstance(value, float)
        assert 0 <= value <= 1

    for name in ["flops", "runtime", "test_loss", "architecture_size", "training_time"]:
        value = getattr(metrics, name)
        assert value is None


def test_architecture_size(evaluator, dummy_CNN_small, dummy_CNN_large):
    small_size = evaluator.compute_architecture_size(dummy_CNN_small)
    large_size = evaluator.compute_architecture_size(dummy_CNN_large)

    assert isinstance(small_size, int)
    assert small_size > 0
    assert isinstance(large_size, int)
    assert large_size > 0
    assert small_size < large_size


def test_invalid_iterations_runtime(evaluator, dummy_CNN_small):
    with pytest.raises(ValueError, match="iterations must be positive"):
        evaluator.compute_runtime(dummy_CNN_small, iterations=-1)
