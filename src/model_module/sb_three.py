from src.model_module.environment import CustomEnv
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
import os


class SBThreeAgent:
    def __init__(
        self,
        policy_algorithm_class: type[BaseAlgorithm],
        policy: type[BasePolicy] | str = "MlpPolicy",
        learning_rate: float = 0.001,
    ):
        self.env: CustomEnv = CustomEnv()
        self.model = policy_algorithm_class(
            policy=policy,
            env=self.env,
            verbose=1,
            device="cpu",
            learning_rate=learning_rate,
        )

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
