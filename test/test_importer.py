import sys
import os
import pytest

# Fixes import issues when running tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_module.dataset import DatasetOption
from src.data_module.importer import DataImporter


def test_get_dataloaders():
    with pytest.raises(ValueError):
        DataImporter(DatasetOption.CIFAR_10).get_dataloaders(batch_size=0)

    assert DataImporter(DatasetOption.CIFAR_10).get_dataloaders(batch_size=1) is not None


def test_get_num_classes():
    assert DataImporter(DatasetOption.EMNIST_BALANCED).get_num_classes() == (47, 47)


def test_get_dimensions():
    assert DataImporter(DatasetOption.EMNIST_LETTERS).get_dimensions() == (1, 28, 28)


"""

OLD TESTS:

def print_image_grid(image_tensor):

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.squeeze(0)
    plt.imshow(image_tensor.numpy(), cmap="gray")
    plt.axis("off")
    plt.show()


def generate_single_image_data():
    # Generate a single 28x28 grayscale image with a white square at a random location
    img_size = (28, 28)
    square_size = 8
    img = torch.zeros(img_size, dtype=torch.float32)
    max_pos = img_size[0] - square_size
    top = random.randint(0, max_pos)
    left = random.randint(0, max_pos)
    img[top : top + square_size, left : left + square_size] = 1.0  # white square at random position
    img = img.view(1, -1)
    return img


def generate_csv_data(num_images=1):  # tiny in-memory CSV for testing
    images = []
    for i in range(num_images):
        label = 0
        img = generate_single_image_data().numpy()[0]
        row = str(label) + "," + ",".join(img.astype(str))
        images.append(row)
    return "\n".join(images)


CSV_DATA_SINGLE_IMAGE = "0," + ",".join(
    row.astype(str) for row in generate_single_image_data().numpy()[0]
)
CSV_DATA = generate_csv_data(num_images=128)

def test_data_imports_correctly(importer):
    assert hasattr(importer, "data")
    assert isinstance(importer.data, torch.Tensor)
    assert importer.data.shape == (128, 1, 28, 28)  # 128 images of shape [1, 28, 28]


@pytest.mark.parametrize("batch_size,test_split", [(16, 0.2), (32, 0.25)])
def test_batch_sizes(importer, batch_size, test_split):
    train_loader, test_loader = importer.get_as_cnn(
        batch_size=batch_size, test_split=test_split, seed=42
    )

    expected_train = math.ceil(128 * (1 - test_split) / batch_size)
    expected_test = math.ceil(128 * test_split / batch_size)

    assert len(train_loader) == expected_train
    assert len(test_loader) == expected_test

    for batch in train_loader:
        images, _ = batch
        assert images.shape[1:] == (1, 28, 28)

def test_batch_size_1(importer):
    batch_size = 16
    train_loader, test_loader = importer.get_as_cnn(batch_size=batch_size, test_split=0.2, seed=42)
    assert len(train_loader) == math.ceil(128 * 0.8 / batch_size)
    assert len(test_loader) == math.ceil(128 * 0.2 / batch_size)
    for batch in train_loader:
        images, _ = batch
        assert images.shape[1:] == (1, 28, 28)


def test_batch_size_2(importer):
    batch_size = 32
    train_loader, test_loader = importer.get_as_cnn(batch_size=batch_size, test_split=0.25, seed=42)
    assert len(train_loader) == math.ceil(128 * 0.75 / batch_size)
    assert len(test_loader) == math.ceil(128 * 0.25 / batch_size)
    for batch in train_loader:
        images, _ = batch
        assert images.shape[1:] == (1, 28, 28) 


@pytest.mark.parametrize("batch_size", [0, -5])
def test_invalid_batch_sizes(importer, batch_size):
    try:
        importer.get_as_cnn(batch_size=batch_size, test_split=0.2, seed=42)
    except ValueError as e:
        assert "batch_size must be a positive integer" in str(e)


@pytest.mark.parametrize("test_split", [0, -5])
def test_invalid_test_split(importer, test_split):
    try:
        train_loader, test_loader = importer.get_as_cnn(
            batch_size=16, test_split=test_split, seed=42
        )
    except ValueError as e:
        assert "test_split must be between 0 and 1 (exclusive)" in str(e)


def test_random_seed_reproducibility(importer):
    batch_size = 16
    seed = 42
    train_loader_1, _ = importer.get_as_cnn(
        batch_size=batch_size, test_split=0.2, seed=seed, shuffle=False
    )
    train_loader_2, _ = importer.get_as_cnn(
        batch_size=batch_size, test_split=0.2, seed=seed, shuffle=False
    )

    for images, _ in train_loader_1:
        specific_image = images[14]  # 4th image in the batch
        break
    for images, _ in train_loader_2:
        specific_image_2 = images[14]  # 4th image in the batch
        break

    assert torch.equal(specific_image, specific_image_2)


def test_different_random_seeds(importer):
    batch_size = 16
    seed1 = 42
    seed2 = 43
    train_loader_1, _ = importer.get_as_cnn(
        batch_size=batch_size, test_split=0.2, seed=seed1, shuffle=False
    )
    train_loader_2, _ = importer.get_as_cnn(
        batch_size=batch_size, test_split=0.2, seed=seed2, shuffle=False
    )

    for images, _ in train_loader_1:
        specific_image_1 = images[14]
        break
    for images, _ in train_loader_2:
        specific_image_2 = images[14]
        break

    assert not torch.equal(specific_image_1, specific_image_2)


def test_shuffle_effect(
    importer,
):  # this test can theoretically fail occasionally, but very unlikely
    batch_size = 16
    train_loader_1, _ = importer.get_as_cnn(
        batch_size=batch_size, test_split=0.2, seed=42, shuffle=True
    )
    train_loader_2, _ = importer.get_as_cnn(
        batch_size=batch_size, test_split=0.2, seed=42, shuffle=True
    )

    for images, _ in train_loader_1:
        specific_image_1 = images[14]
        break
    for images, _ in train_loader_2:
        specific_image_2 = images[14]
        break

    assert not torch.equal(specific_image_1, specific_image_2)


# test batch size gives correct number of batches
# test train/test split sizes
# test random seed is working
"""
