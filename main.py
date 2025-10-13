from src.model_module.sb_three import SBThreeAgent
from rich.console import Console
from stable_baselines3 import PPO  # Add this import


console = Console()


def main():
    console.print("[bold blue]Initializing NAS RL Agent...[/bold blue]")

    # Initialize agent with PPO algorithm
    agent = SBThreeAgent(policy_algorithm_class=PPO)

    # Train the agent (reduced for testing reward changes)
    console.print("[bold green]Starting training...[/bold green]")
    agent.train(total_timesteps=500)  # Reduced to see reward patterns faster

    # Save the trained model
    agent.save_model()

    # Evaluate the trained agent
    console.print("[bold yellow]Evaluating trained agent...[/bold yellow]")
    agent.evaluate(num_episodes=5)


if __name__ == "__main__":
    main()
