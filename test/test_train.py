import torch
from src.environment.train import Trainer
from src.utils.data_importer.importer import DataImporter
from src.utils.data_importer.dataset import DatasetOption


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
        loss_function=torch.nn.CrossEntropyLoss(),
    )

    stopped_while_trianing = trainer.train(model, optimizer, 1)
    assert stopped_while_trianing is True

    stopped_while_trianing = trainer.train(model, optimizer)
    assert stopped_while_trianing is False