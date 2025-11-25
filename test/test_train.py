import torch
from src.classification_module.train import Trainer
from src.data_module.importer import DataImporter
from src.data_module.dataset import DatasetOption
from src.environment.metrics import Metrics


def test_stop_when_trains_too_long():
    importer = DataImporter(dataset_option=DatasetOption.EMNIST_LETTERS)
    dataloaders = importer.get_dataloaders(batch_size=64, shuffle=True)
    number_of_classes = 26
    pre_model = [
        torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Flatten(),
        torch.nn.Linear(32 * 13 * 13, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, number_of_classes),
        torch.nn.Softmax(dim=1),
    ]
    model = torch.nn.Sequential(
        *pre_model,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    trainer = Trainer(
        dataloaders=dataloaders,
        model=model,
        optimizer=optimizer,
        loss_function=torch.nn.CrossEntropyLoss(),
        num_classes=number_of_classes,
        dimensions=(1, 28, 28),
    )

    stopped_while_trianing = trainer.train(1)
    assert stopped_while_trianing is True

    stopped_while_trianing = trainer.train()
    assert stopped_while_trianing is False

    test_return_metrics = trainer.test()
    assert isinstance(test_return_metrics, Metrics)
