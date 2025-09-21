import torch
import pytest
from src.importer import (
    DataImporter,
    ConvolutionalArguments,
    GRAYSCALE_NUM_CHANNELS,
)


def generate_single_image_data():
    # Generate a single 28x28 grayscale image with a simple pattern
    img_size = (28, 28)
    img = torch.zeros(img_size, dtype=torch.float32)
    img[10:18, 10:18] = 1.0  # white square in the center
    img = img.view(1, -1)  # flatten to [1, 784]
    return img


# Create a tiny in-memory CSV for testing
CSV_DATA = "0," + ",".join(row.astype(str) for row in generate_single_image_data().numpy()[0])


@pytest.fixture
def importer(tmp_path):
    # write small CSV to temp file
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(CSV_DATA)
    return DataImporter(filepath=str(csv_file), img_size=(28, 28))


def test_ffnn_shape(importer):
    data = importer.get_as_ffnn()
    assert isinstance(data, torch.Tensor)
    assert data.shape[1] == 28 * 28


def test_cnn_no_batch(importer):
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
