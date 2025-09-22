import sys
import os
import torch
import pytest

# Ensure the project root is in sys.path so 'src' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_module.importer import (
    DataImporter,
)


def generate_single_image_data():
    # Generate a single 28x28 grayscale image with a simple pattern
    img_size = (28, 28)
    img = torch.zeros(img_size, dtype=torch.float32)
    img[10:18, 10:18] = 1.0  # white square in the center
    img = img.view(1, -1)  # flatten to [1, 784]
    return img


# Create a tiny in-memory CSV for testing
def generate_csv_data(num_images=1):
    images = []
    for i in range(num_images):
        label = 0  # or use i % 10 for multiple classes
        img = generate_single_image_data().numpy()[0]
        row = str(label) + "," + ",".join(img.astype(str))
        images.append(row)
    return "\n".join(images)


CSV_DATA_SINGLE_IMAGE = "0," + ",".join(
    row.astype(str) for row in generate_single_image_data().numpy()[0]
)
CSV_DATA = generate_csv_data(num_images=128)


@pytest.fixture
def importer(tmp_path):
    # write small CSV to temp file
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(CSV_DATA)
    return DataImporter(filepath=str(csv_file), img_size=(28, 28))


def test_ffnn_shape(importer):
    train_dataloader, test_dataloader = importer.get_as_ffnn(batch_size=6, test_split=0.1)
    batches = list(train_dataloader)
    for i, (X, y) in enumerate(batches):
        if i < len(batches) - 1:
            assert X.shape == (6, 1, 28, 28)
            assert y.shape == (6,)
        else:
            assert X.shape == (2, 1, 28, 28)
            assert y.shape == (2,)

    train_dataloader, test_dataloader = importer.get_as_ffnn(batch_size=20, test_split=0.2)
    batches = list(test_dataloader)
    for i, (X, y) in enumerate(batches):
        if i < len(batches) - 1:
            assert X.shape == (20, 1, 28, 28)
            assert y.shape == (20,)
        else:
            assert X.shape == (5, 1, 28, 28)
            assert y.shape == (5,)

    train_dataloader, test_dataloader = importer.get_as_ffnn(batch_size=128, test_split=0.2)
    batches = list(test_dataloader)
    for i, (X, y) in enumerate(batches):
        if i < len(batches) - 1:
            assert X.shape == (128, 1, 28, 28)
            assert y.shape == (128,)
        else:
            assert X.shape == (25, 1, 28, 28)
            assert y.shape == (25,)


def test_ffnn_no_batch(importer):
    train_dataloader, test_dataloader = importer.get_as_ffnn(batch_size=None, test_split=0.1)
    for X, y in train_dataloader:
        print("y ", y)
        assert X.shape == (1, 1, 28, 28)
        assert y.shape == (1,)

    train_dataloader, test_dataloader = importer.get_as_ffnn(batch_size=-871263, test_split=0.1)
    for X, y in test_dataloader:
        assert X.shape == (1, 1, 28, 28)
        assert y.shape == (1,)


""" def test_cnn_no_batch(importer):
    cnn_data = importer.get_as_cnn()
    assert isinstance(cnn_data, torch.Tensor)
    assert cnn_data.shape[1:] == (GRAYSCALE_NUM_CHANNELS, 28, 28)


def test_cnn_with_batch(importer):
    batch_size = 1
    cnn_batches = importer.get_as_cnn(batch_size=batch_size)
    assert isinstance(cnn_batches, tuple)
    for batch in cnn_batches:
        assert batch.shape[1:] == (GRAYSCALE_NUM_CHANNELS, 28, 28)


def test_cnn_with_conv(importer):
    conv_args = ConvolutionalArguments(in_channels=1, out_channels=2, kernel_size=2)
    cnn_data, conv = importer.get_as_cnn(conv_args=conv_args)
    assert isinstance(cnn_data, torch.Tensor)
    assert isinstance(conv, torch.nn.Conv2d)

    # sanity: convolve a small batch
    out = conv(cnn_data)
    assert out.shape[1] == conv_args.out_channels


def test_cnn_with_batch_and_conv(importer):
    conv_args = ConvolutionalArguments(in_channels=1, out_channels=2, kernel_size=2)
    batched_data, conv = importer.get_as_cnn(batch_size=1, conv_args=conv_args)
    all(isinstance(t, torch.Tensor) for t in batched_data)  # check each element
    assert isinstance(conv, torch.nn.Conv2d)
    out_batches = [conv(batch) for batch in batched_data]
    print(out_batches[0].shape)
    for out in out_batches:
        assert (
            out.shape[0] == conv_args.out_channels
        )  # out channels is first index because data is only a single image
 """
