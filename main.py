import os
from rich.console import Console
from stable_baselines3 import PPO  # Add this import
import warnings
import shutil

from src.model_module.sb_three import SBThreeAgent

warnings.filterwarnings("ignore", message="Unsupported operator aten::tanh")
console = Console()


def main():
    console.print("[bold blue]Initializing Neural Architecture Search...[/bold blue]")

    # Clean up old models
    if os.path.exists("saved_models"):
        shutil.rmtree("saved_models")
        console.print("[yellow]Deleted old saved models[/yellow]")

    # Initialize agent with PPO algorithm
    agent = SBThreeAgent(policy_algorithm_class=PPO)

    # Train the agent
    console.print("[bold green]Starting training...[/bold green]")
    agent.train(total_timesteps=30, log_interval=1)

    # Save the trained model
    agent.save_model()

    # Evaluate the trained agent
    console.print("[bold yellow]Evaluating trained agent...[/bold yellow]")
    agent.evaluate(num_episodes=5)


if __name__ == "__main__":
    main()
