import os
from rich.console import Console  # Add this import
import warnings
import shutil
import argparse
import torch

from src.model_module.sb_three import SBThreeAgent
from src.data_module.importer import DatasetOption
from sb3_contrib.ppo_mask import MaskablePPO

from src.action_masking.action_masking_policy import CustomMaskablePolicy

warnings.filterwarnings("ignore", message="Unsupported operator aten::tanh")
console = Console()


def main():
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize-hyperparameters", type=int, default=5, help="If set, runs hyperparameter optimization with given number of trials")
    parser.add_argument("--clean-saved-models", action="store_true")
    parser.add_argument("--policy-seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--torch-seed", type=int, default=None, help="Random seed for classifier initialization")
    parser.add_argument("--optuna-seed", type=int, default=None, help="Random seed for hyperparameter optimization")

    args = parser.parse_args()

    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)
        console.print(f"[yellow]Set torch seed to '{args.torch_seed}'[/yellow]")

    if args.policy_seed is not None:
        console.print(f"[yellow]Set policy seed to '{args.policy_seed}'[/yellow]")

    if args.optuna_seed is not None:
        console.print(f"[yellow]Set optuna seed to '{args.optuna_seed}'[/yellow]")

    if args.optimize_hyperparameters:
        from src.model_module.hyperparameter_tuning import (
            SLHyperparameterOptimizer,
            HyperparameterSearchSpace,
        )

        console.print("[bold magenta]Starting Hyperparameter Optimization...[/bold magenta]")
        optimizer = SLHyperparameterOptimizer(search_space=HyperparameterSearchSpace(), n_trials=args.optimize_hyperparameters, seed=args.optuna_seed)
        best_hyperparameters = optimizer.optimize()
        console.print(f"[bold magenta]Best Hyperparameters: \n{best_hyperparameters}[/bold magenta]")
        return

    console.print("[bold blue]Initializing Neural Architecture Search...[/bold blue]")

    # Clean up old models
    if args.clean_saved_models and os.path.exists("saved_models"):
        shutil.rmtree("saved_models")
        console.print("[yellow]Deleted old saved models[/yellow]")

    # Initialize agent with PPO algorithm
    agent = SBThreeAgent(
        policy_algorithm_class=MaskablePPO, policy=CustomMaskablePolicy, policy_seed=args.policy_seed
    )

    # Train the agent
    console.print("[bold green]Starting training...[/bold green]")
    agent.train(total_timesteps=30)

    # Save the trained model
    agent.save_model()

    # Evaluate the trained agent
    console.print("[bold yellow]Evaluating trained agent...[/bold yellow]")
    agent.evaluate(num_episodes=5)


if __name__ == "__main__":
    main()
