import os
from src.model_module.sb_three import SBThreeAgent
from rich.console import Console
from stable_baselines3 import PPO  # Add this import

console = Console()


def main():
    console.print("[bold blue]Initializing NAS RL Agent...[/bold blue]")

    # Delete old models to force fresh start
    import shutil

    if os.path.exists("saved_models"):
        shutil.rmtree("saved_models")
        console.print("[yellow]Deleted old saved models[/yellow]")

    # Initialize agent with PPO algorithm
    agent = SBThreeAgent(policy_algorithm_class=PPO)

    # Train the agent
    console.print("[bold green]Starting training...[/bold green]")
    agent.train(total_timesteps=500)

    # Save the trained model
    agent.save_model()

    # Evaluate the trained agent
    console.print("[bold yellow]Evaluating trained agent...[/bold yellow]")
    agent.evaluate(num_episodes=5)


if __name__ == "__main__":
    main()
