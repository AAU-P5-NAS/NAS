import os
from rich.console import Console  # Add this import
import shutil
from sb3_contrib.ppo_mask import MaskablePPO
import torch

from src.agent.agent import RLAgent
from src.agent.action_masking.action_masking_policy import CustomMaskablePolicy

console = Console()


def main():
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        console.print("[bold green]CUDA is available. Using GPU for training.[/bold green]")
    else:
        console.print("[bold red]CUDA is not available. Using CPU for training.[/bold red]")
    console.print("[bold blue]Initializing Neural Architecture Search...[/bold blue]")

    # Clean up old models
    if os.path.exists("saved_models"):
        shutil.rmtree("saved_models")
        console.print("[yellow]Deleted old saved models[/yellow]")

    # Initialize agent with PPO algorithm
    agent = RLAgent(
        policy_algorithm_class=MaskablePPO,
        policy=CustomMaskablePolicy,
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
