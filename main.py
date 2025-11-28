import argparse
import os
from rich.console import Console  # Add this import
import shutil
from sb3_contrib.ppo_mask import MaskablePPO
import torch

from src.environment.reward import Weights
from src.agent.agent import RLAgent
from src.agent.action_masking.action_masking_policy import CustomMaskablePolicy

console = Console()


def log_cuda_mem(tag=""):
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    device = torch.cuda.current_device()

    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    free, total = torch.cuda.mem_get_info()  # free/total *actual* device memory
    free /= 1024**2
    total /= 1024**2

    print(f"\n=== {tag} ===")
    print(f"Allocated by PyTorch: {allocated:.2f} MiB")
    print(f"Reserved by PyTorch:  {reserved:.2f} MiB")
    print(f"GPU Free:             {free:.2f} MiB")
    print(f"GPU Total:            {total:.2f} MiB")


def main():
    log_cuda_mem()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimize-hyperparameters",
        type=int,
        default=None,
        help="If set, runs hyperparameter optimization with given number of trials",
    )
    parser.add_argument("--clean-saved-models", action="store_true")
    parser.add_argument(
        "--policy-seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--torch-seed", type=int, default=None, help="Random seed for classifier initialization"
    )
    parser.add_argument(
        "--optuna-seed", type=int, default=None, help="Random seed for hyperparameter optimization"
    )

    args = parser.parse_args()

    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)
        console.print(f"[yellow]Set torch seed to '{args.torch_seed}'[/yellow]")

    if args.policy_seed is not None:
        console.print(f"[yellow]Set policy seed to '{args.policy_seed}'[/yellow]")

    if args.optuna_seed is not None:
        console.print(f"[yellow]Set optuna seed to '{args.optuna_seed}'[/yellow]")

    for arg in vars(args):
        console.print(f"[blue]{arg}: {getattr(args, arg)}[/blue]")
    if args.optimize_hyperparameters is not None:
        from src.utils.hyperparameter_tuning import (
            SLHyperparameterOptimizer,
            HyperparameterSearchSpace,
        )

        console.print("[bold magenta]Starting Hyperparameter Optimization...[/bold magenta]")
        optimizer = SLHyperparameterOptimizer(
            search_space=HyperparameterSearchSpace(),
            n_trials=args.optimize_hyperparameters,
            seed=args.optuna_seed,
        )
        best_hyperparameters = optimizer.optimize()
        console.print(
            f"[bold magenta]Best Hyperparameters: \n{best_hyperparameters}[/bold magenta]"
        )
        return

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        console.print("[bold green]CUDA is available. Using GPU for training.[/bold green]")
    else:
        console.print("[bold red]CUDA is not available. Using CPU for training.[/bold red]")
    console.print("[bold blue]Initializing Neural Architecture Search...[/bold blue]")

    # Clean up old models
    if args.clean_saved_models and os.path.exists("saved_models"):
        shutil.rmtree("saved_models")
        console.print("[yellow]Deleted old saved models[/yellow]")

    # Initialize agent with PPO algorithm
    agent = RLAgent(
        policy_algorithm_class=MaskablePPO,
        policy=CustomMaskablePolicy,
        policy_seed=args.policy_seed,
        reward_weights=Weights(accuracy=0.9, flops=0.1),
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
