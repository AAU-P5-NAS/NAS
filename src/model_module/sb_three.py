import torch
from src.model_module.environment import CustomEnv
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from src.classification_module.reward import Weights
import os


class SBThreeAgent:
    def __init__(
        self,
        policy_algorithm_class: type[BaseAlgorithm],
        policy: type[BasePolicy] = ActorCriticPolicy,
        learning_rate: float = 0.001,
        training_epochs: int = 15,
        arch_learning_rate: float = 0.001,
        arch_momentum: float = 0.9,
        batch_size: int = 64,
        reward_weights: Weights | None = None,
    ):
        self.env: CustomEnv = CustomEnv(
            training_epochs=training_epochs,
            arch_learning_rate=arch_learning_rate,
            arch_momentum=arch_momentum,
            batch_size=batch_size,
            reward_weights=reward_weights,
        )
        self.model = policy_algorithm_class(
            policy=policy,
            env=self.env,
            verbose=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            learning_rate=learning_rate,
        )
        print(next(self.model.policy.parameters()).device)  # should output cuda:0

    def train(self, total_timesteps: int = 10000):
        self.model.learn(total_timesteps=total_timesteps)

    def save_model(self, path: str = "saved_models/ppo_agent"):
        """Save the trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path: str = "saved_models/ppo_agent"):
        """Load a previously trained model"""
        if os.path.exists(f"{path}.zip"):
            self.model = self.model.load(path, env=self.env)
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}")

    def evaluate(self, num_episodes: int = 10):
        """Evaluate the trained agent"""
        total_rewards = []

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated

            total_rewards.append(episode_reward)

        avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")
        return avg_reward
