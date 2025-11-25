import torch
from src.utils.logger import TensorboardLogger
from src.utils.hyperparameters import SLHyperParameters
from src.environment.environment import CustomEnv
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy

from sb3_contrib.common.wrappers import ActionMasker

from src.classification_module.reward import WeightedSumRS, Weights
import os

from stable_baselines3.common.logger import Logger
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.logger import TensorBoardOutputFormat

device = "cuda" if torch.cuda.is_available() else "cpu"


def mask_fn(env):
    return env.get_action_mask()


hyperparameters = SLHyperParameters(
    training_epochs=15,
    learning_rate=0.001,
    momentum=0.9,
    batch_size=64,
)

# Tensorboard logging setup
log_folder = "tensorboard_logs/"
log_interval = 1  # log every n steps
num_existing_logs = len(
    [name for name in os.listdir(log_folder) if os.path.isdir(os.path.join(log_folder, name))]
)
run_name = f"NAS_run {num_existing_logs}"
logger = Logger(
    folder=log_folder, output_formats=[TensorBoardOutputFormat(f"{log_folder}/{run_name}")]
)
writer = SummaryWriter(log_dir=f"{log_folder}/{run_name}")
tb_logger = TensorboardLogger(logger=logger, writer=writer, log_folder=log_folder)  # created once


class RLAgent:
    TB_LOG_NAME: str = "RLAgent_run"
    TB_LOG_DIRECTORY: str = "tensorboard_logs/"
    MODEL_SAVE_DIRECTORY: str = "saved_models/"
    model_save_path: str

    def __init__(
        self,
        policy_algorithm_class: type[BaseAlgorithm],
        policy: type[BasePolicy] | str = "MlpPolicy",
        rl_learning_rate: float = 0.001,
        hyperparameters: SLHyperParameters = hyperparameters,
        reward_weights: Weights | None = None,
    ):
        self.env = ActionMasker(
            CustomEnv(
                device=device,
                hyperparameters=hyperparameters,
                tb_logger=tb_logger,
                reward_strategy=WeightedSumRS(weights=reward_weights)
                if reward_weights
                else WeightedSumRS(Weights(accuracy=0.5, flops=0.5)),
            ),
            action_mask_fn=mask_fn,
        )
        self.model = policy_algorithm_class(
            policy=policy,
            env=self.env,
            verbose=1,
            gamma=1,  # type: ignore # extremely important to have gamma=1 for maximum discount
            device="cpu",
            learning_rate=rl_learning_rate,
        )
        self.model.set_logger(tb_logger.logger)
        self.model_save_path = f"{self.MODEL_SAVE_DIRECTORY}{self.model.__class__.__name__}"

        print(next(self.model.policy.parameters()).device)  # should output cuda:0

        self.check_directories()

    def train(self, total_timesteps: int = 10000):
        self.model.learn(
            total_timesteps=total_timesteps,
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
                episode_reward += reward  # type: ignore
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
