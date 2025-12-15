
import torch
from rich.console import Console
from src.environment.train import Trainer
from torch import nn
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
    best_complexy_arch = unflatten_cnn_config(sortedArchs.complexity_sorted[0].arch, 7)
    console.print("complexity" + get_layers_as_str(best_complexy_arch.layers, True))
    best_snip_arch = unflatten_cnn_config(sortedArchs.snip_sorted[0].arch, 7)
    console.print("snip" + get_layers_as_str(best_snip_arch.layers, True))
    best_synflow_arch = unflatten_cnn_config(sortedArchs.synflow_sorted[0].arch, 7)
    console.print("synflow" + get_layers_as_str(best_synflow_arch.layers, True))
    best_jacov_arch = unflatten_cnn_config(sortedArchs.jacov_sorted[0].arch, 7)
    console.print("jacov" + get_layers_as_str(best_jacov_arch.layers, True))
    best_we_arch = unflatten_cnn_config(sortedArchs.ws_sorted[0].arch, 7)
    console.print("weighted" + get_layers_as_str(best_we_arch.layers, True))
    


    best_complexy_arch = Architecture(best_complexy_arch, 10, (3, 32, 32))
    best_snip_arch =  Architecture(best_snip_arch, 10, (3, 32, 32))
    best_synflow_arch =  Architecture(best_synflow_arch, 10, (3, 32, 32))
    best_jacov_arch =  Architecture(best_jacov_arch, 10, (3, 32, 32))
    best_we_arch =  Architecture(best_we_arch, 10, (3, 32, 32))
    best_arch = [#(best_complexy_arch, "best_complexy_arch"), 
                 #(best_snip_arch, "best_snip_arch"),
                 #(best_synflow_arch, "best_synflow_arch"),
                 #(best_jacov_arch, "best_jacov_arch"),
                 (best_we_arch, "best_we_arch")]
    
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
    
        for epoch in range(100):
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



