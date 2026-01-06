import argparse
import os
from rich.console import Console  # Add this import
import shutil
from stable_baselines3 import PPO
import torch

from src.environment.reward.tchebycheff import TchebycheffRS
from src.environment.reward.weighted_sum import WeightedSumRS
from src.environment.reward.archive_pareto_dom import DominanceNoveltyRS
from src.environment.reward.reward import Weights
from src.agent.agent import RLAgent
from src.agent.action_masking.action_masking_policy import CustomMaskablePolicy
from src.utils import email
from src.utils.arguments import ParseArguments
from src.utils.architecture import unflatten_cnn_config

console = Console()


def get_agent(args: argparse.Namespace) -> RLAgent:
    weights = Weights(accuracy=0.8, flops=0.2)
    if args.evaluate_archive:
        reward_strategy = WeightedSumRS(weights)
    elif args.use_tchebycheff:
        reward_strategy = TchebycheffRS(weights)
    elif args.use_dominance_novelty:
        reward_strategy = DominanceNoveltyRS(weights)
    elif args.use_weighted_sum:
        reward_strategy = WeightedSumRS(weights)
    else:  # Default to Weighted Sum
        reward_strategy = WeightedSumRS(weights)

    # Initialize the RL agent
    agent = RLAgent(
        policy_algorithm_class=PPO,
        policy=CustomMaskablePolicy,
        policy_seed=args.policy_seed,
        reward_strategy=reward_strategy,
    )

    if args.load_model is not None:
        # Load an existing model into the agent
        console.print(f"[bold green]Loading model '{args.load_model}'[/bold green]")
        agent.load_model(args.load_model)

    return agent


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
        agent = get_agent(args)

        if args.evaluate_archive:
            console.print("[bold green]Loading archive from[/bold green]")
            loader = DominanceNoveltyRS(Weights(accuracy=0.8, flops=0.2))
            archive = loader.elite_archive.load_archive()
            if archive is None:
                console.print("[bold red]No archive found.[/bold red]")
                return
            console.print("[bold yellow]Evaluating archive...[/bold yellow]")
            for entry in archive:
                agent.env.evaluate_architecture(
                    unflatten_cnn_config(entry.arch, max_layers=7), log_arch=True
                )
            return

        # Train the agent
        console.print("[bold green]Starting training...[/bold green]")
        agent.train(total_timesteps=1000000)

        # Save the trained model
        agent.save_model()

        # Evaluate the trained agent
        console.print("[bold yellow]Evaluating trained agent...[/bold yellow]")
        agent.evaluate(num_episodes=5)

    except Exception as e:
        print("An error occurred during training or evaluation:", e)
        if args.report_exception:
            email.ReportException(exception=e)
        agent.save_model()
        if isinstance(agent.env.reward_strategy, DominanceNoveltyRS):
            agent.env.reward_strategy.elite_archive.save_archive()

    except KeyboardInterrupt:
        console.print("[bold red]Training interrupted by user.[/bold red]")
        agent.save_model()
        if isinstance(agent.env.reward_strategy, DominanceNoveltyRS):
            agent.env.reward_strategy.elite_archive.save_archive()


if __name__ == "__main__":
    main()
