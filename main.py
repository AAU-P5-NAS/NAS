from src.model_module.sb_three import SBThreeAgent
from rich.console import Console
from stable_baselines3 import PPO  # Add this import


console = Console()


def main():
    # initialize agent with PPO algorithm

    model = SBThreeAgent(policy_algorithm_class=PPO)

    model.train(total_timesteps=10)


if __name__ == "__main__":
    main()
