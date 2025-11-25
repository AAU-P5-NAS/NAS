import os
from rich.console import Console  # Add this import
import warnings
import shutil
from sb3_contrib.ppo_mask import MaskablePPO

from model_module.agent import RLAgent
from src.action_masking.action_masking_policy import CustomMaskablePolicy

warnings.filterwarnings("ignore", message="Unsupported operator aten::tanh")
console = Console()


def main():
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
