import torch
from src.model_module.environment import CustomEnv
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import TensorBoardOutputFormat
from sb3_contrib.common.wrappers import ActionMasker

from src.classification_module.reward import WeightedSumRS, Weights
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


def mask_fn(env):
    return env.get_action_mask()


class CustomEnvCallBack(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomEnvCallBack, self).__init__(verbose)

    def _on_training_start(self):
        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tensorboard_formatter = next(
            formatter
            for formatter in output_formats
            if isinstance(formatter, TensorBoardOutputFormat)
        )

    def _on_step(self) -> bool:
        if isinstance(self.training_env, DummyVecEnv):
            for env in self.training_env.envs:
                if isinstance(env, Monitor):
                    if isinstance(env.env, CustomEnv):
                        env.env._write_summary(self.logger, self.tensorboard_formatter.writer)
        return True


class SBThreeAgent:
    TB_LOG_NAME: str = "SBThreeAgent_run"
    TB_LOG_DIRECTORY: str = "tensorboard_logs/"
    MODEL_SAVE_DIRECTORY: str = "saved_models/"
    model_save_path: str

    def __init__(
        self,
        policy_algorithm_class: type[BaseAlgorithm],
        policy: type[BasePolicy] | str = "MlpPolicy",
        learning_rate: float = 0.001,
        training_epochs: int = 15,
        arch_learning_rate: float = 0.001,
        arch_momentum: float = 0.9,
        batch_size: int = 64,
        reward_weights: Weights | None = None,
        showSamples: bool = False,
    ):
        self.env = ActionMasker(
            CustomEnv(
                device=device,
                logdir=self.TB_LOG_DIRECTORY,
                training_epochs=training_epochs,
                arch_learning_rate=arch_learning_rate,
                arch_momentum=arch_momentum,
                batch_size=batch_size,
                reward_strategy=WeightedSumRS(weights=reward_weights)
                if reward_weights
                else WeightedSumRS(Weights(accuracy=0.5, flops=0.5)),
                showSamples=showSamples,
            ),
            action_mask_fn=mask_fn,
        )
        self.model = policy_algorithm_class(
            policy=policy,
            env=self.env,
            verbose=1,
            gamma=1,  # type: ignore # extremely important to have gamma=1 for maximum discount
            device="cpu",
            learning_rate=learning_rate,
            tensorboard_log=self.TB_LOG_DIRECTORY,
        )
        self.model_save_path = f"{self.MODEL_SAVE_DIRECTORY}{self.model.__class__.__name__}"

        print(next(self.model.policy.parameters()).device)  # should output cuda:0

        self.check_directories()

    def train(self, total_timesteps: int = 10000, log_interval: int = 1):
        self.model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=self.TB_LOG_NAME,
            log_interval=log_interval,
            callback=CustomEnvCallBack(),
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
