import torch
from rich.console import Console
from src.environment.train import Trainer
from src.environment.metrics import Evaluator
from src.utils.data_importer.dataset import DatasetOption
from src.utils.data_importer.importer import DataImporter
from torch.optim import Adam
from src.environment.reward.archive_pareto_dom import ElitistArchive
from src.utils.architecture import unflatten_cnn_config, Architecture
from src.utils.logger import get_layers_as_str


console = Console()


def main():
    torch.manual_seed(42)
    sortedArchs = ElitistArchive().sort_archs()

    # Extract top 3 architectures for each category
    categories = [
        ("complexity", sortedArchs.complexity_sorted),
        ("snip", sortedArchs.snip_sorted),
        ("synflow", sortedArchs.synflow_sorted),
        ("jacov", sortedArchs.jacov_sorted),
        ("weighted", sortedArchs.ws_sorted),
    ]

    best_arch = []
    for cat_name, arch_list in categories:
        for i in range(3):
            arch_cfg = unflatten_cnn_config(arch_list[i].arch, 7)
            console.print(f"{cat_name} {i + 1}: " + get_layers_as_str(arch_cfg.layers, True))
            arch = Architecture(arch_cfg, 10, (3, 32, 32))
            best_arch.append((arch, f"{cat_name}_{i + 1}"))

    validated_metrics: dict[str, list] = {}
    for model, name in best_arch:
        metrics = [None, None, None]
        importer = DataImporter(dataset_option=DatasetOption.CIFAR_10)

        trainer = Trainer(
            dataloaders=importer.get_dataloaders(batch_size=64),
            loss_function=torch.nn.CrossEntropyLoss(),
        )

        evaluator = Evaluator(
            num_classes=10,
            dataloaders=importer.get_dataloaders(batch_size=64, shuffle=False),
            dimensions=importer.get_dimensions(),
            device=torch.device("cuda"),
            loss_function=torch.nn.CrossEntropyLoss(),
        )
        optimizer = Adam(model.parameters(), 0.00132)

        for epoch in range(30):
            print("Epoch:", epoch + 1)
            trainer.train(model, optimizer)
            metrics = evaluator.evaluate(model)
            console.print(
                f"[bold green]{name}, Metrics: Accuracy: {metrics.accuracy:.4f}, FLOPS: {metrics.flops:.2f}[/bold green]"
            )

            if validated_metrics.get(name) is None or validated_metrics[name][0] < metrics.accuracy:
                validated_metrics[name] = [metrics.accuracy, metrics.flops]

    console.print("\n[bold yellow]Final validated metrics:[/bold yellow]")
    for name, metric in validated_metrics.items():
        console.print(
            f"[bold green]{name}, Metrics: Accuracy: {metric[0]:.4f}, FLOPS: {metric[1]:.2f}[/bold green]"
        )


if __name__ == "__main__":
    main()
