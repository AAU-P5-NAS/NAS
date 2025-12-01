import os
from rich.console import Console  # Add this import
import shutil
from stable_baselines3 import PPO
import torch

from src.environment.reward import Weights
from src.agent.agent import RLAgent
from src.agent.action_masking.action_masking_policy import CustomMaskablePolicy
from src.utils import email
from src.utils.arguments import ParseArguments

console = Console()


def main():

    args = ParseArguments()

    try:   
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
            policy_algorithm_class=PPO,
            policy=CustomMaskablePolicy,
            policy_seed=args.policy_seed,
            reward_weights=Weights(accuracy=0.90, flops=0.10),
        )

        # Train the agent
        console.print("[bold green]Starting training...[/bold green]")
        agent.train(total_timesteps=50000)

        # Save the trained model
        agent.save_model()

        # Evaluate the trained agent
        console.print("[bold yellow]Evaluating trained agent...[/bold yellow]")
        agent.evaluate(num_episodes=5)
        
    except Exception as e:
        if args.report_exception:
            email.ReportException(exception=e)
        agent.save_model()


        

if __name__ == "__main__":
    main()
