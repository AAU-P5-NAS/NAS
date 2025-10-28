import os

from src.model_module.staged_training import MultiStageTrainer
from rich.console import Console
import warnings
import shutil

warnings.filterwarnings("ignore", message="Unsupported operator aten::tanh")
console = Console()


def main():
    console.print("[bold blue]Initializing NAS Multi-Stage Training...[/bold blue]")

    # Clean up old models
    if os.path.exists("saved_models"):
        shutil.rmtree("saved_models")
        console.print("[yellow]Deleted old saved models[/yellow]")

    os.makedirs("saved_models", exist_ok=True)

    # Initialize multi-stage trainer
    trainer = MultiStageTrainer()

    # Run iterative multi-stage training:
    # Architecture Search → Hyperparameter Optimization → (Architecture → Hyperparams)* → until convergence
    results = trainer.run_all_stages(
        stage1_timesteps=50000,  # Initial architecture search with default hyperparameters
        stage2_timesteps=10000,  # Hyperparameter optimization (shorter for faster trials)
        stage2_trials=10,  # Number of optimization trials per iteration
        stage3_timesteps=50000,  # Architecture search per iteration
        max_iterations=5,  # Maximum number of iterations
        improvement_threshold=0.001,  # Minimum improvement to consider significant
        no_improvement_limit=2,  # Stop after N iterations without improvement
    )

    console.print("\n[bold green]All stages complete![/bold green]")
    console.print(f"[bold cyan]Final best reward: {results['best_reward']:.4f}[/bold cyan]")
    console.print(f"[bold cyan]Total iterations: {results['total_iterations']}[/bold cyan]")

    if results["converged"]:
        console.print(
            "[bold green]✓ Training converged (no improvement for multiple iterations)[/bold green]"
        )
    else:
        console.print("[bold yellow]Training stopped at max iterations[/bold yellow]")


if __name__ == "__main__":
    main()
