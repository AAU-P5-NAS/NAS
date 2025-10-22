import torch
from src.model_module.environment import CustomEnv
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy

import os


class SBThreeAgent:
    TB_LOG_NAME: str = "SBThreeAgent_run"
    TB_LOG_DIRECTORY: str = "tensorboard_logs/"
    MODEL_SAVE_DIRECTORY: str = "saved_models/"
    model_save_path: str

    def __init__(
        self,
        policy_algorithm_class: type[BaseAlgorithm],
        policy: type[BasePolicy] = ActorCriticPolicy,
        learning_rate: float = 0.001,
    ):
        self.env: CustomEnv = CustomEnv()
        self.model = policy_algorithm_class(
            policy=policy,
            env=self.env,
            verbose=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            learning_rate=learning_rate,
            tensorboard_log=self.TB_LOG_DIRECTORY,
        )
        self.model_save_path = f"{self.MODEL_SAVE_DIRECTORY}{self.model.__class__.__name__}"

        print(next(self.model.policy.parameters()).device)  # should output cuda:0

        self.check_directories()

    def train(self, total_timesteps: int = 10000):
        self.model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=self.TB_LOG_NAME,
            log_interval=1,
        )

    def save_model(self):
        """Save the trained model"""
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        self.model.save(self.model_save_path)
        print(f"Model saved to '{self.model_save_path}'")

    def load_model(self):
        """Load a previously trained model"""
        if os.path.exists(f"{self.model_save_path}.zip"):
            self.model = self.model.load(self.model_save_path, env=self.env)
            print(f"Model loaded from '{self.model_save_path}'")
        else:
            print(f"No model found at '{self.model_save_path}'")

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

    def check_directories(self):
        """Check and create necessary directories"""
        os.makedirs(os.path.dirname(self.TB_LOG_DIRECTORY), exist_ok=True)
        os.makedirs(os.path.dirname(self.MODEL_SAVE_DIRECTORY), exist_ok=True)

        print(f"Model will be saved to '{self.model_save_path}'")
        print(f"TensorBoard logs will be saved to '{self.TB_LOG_DIRECTORY}'")
