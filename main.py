import os

import torch
from src.model_module.sb_three import SBThreeAgent
from rich.console import Console
from stable_baselines3 import A2C  # Add this import

print("torch.version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device count:", torch.cuda.device_count())
    print("current device:", torch.cuda.current_device())
    try:
        print("device name:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("get_device_name failed:", e)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

console = Console()


def main():
    console.print("[bold blue]Initializing NAS RL Agent...[/bold blue]")

    # Initialize agent with PPO algorithm
    agent = SBThreeAgent(policy_algorithm_class=A2C)

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
