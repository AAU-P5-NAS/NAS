from typing import Optional
import numpy as np
import torch
from src.utils.logger import TensorboardLogger
from src.utils.hyperparameters import SLHyperParameters
from src.environment.environment import CustomEnv
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy


from src.environment.reward import WeightedSumRS, Weights
import os

from stable_baselines3.common.logger import Logger
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.logger import TensorBoardOutputFormat

from stable_baselines3.common.callbacks import BaseCallback


class EpisodeLimitCallback(BaseCallback):
    def __init__(self, max_episodes: int, verbose=0):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.episode_count = 0

    def _on_step(self) -> bool:
        print("number of steps:", self.num_timesteps)
        print("number of episodes:", self.episode_count)
        infos = self.locals.get("infos", [])
        for info in infos:
            # SB3 injects 'episode' key into infos at the end of each episode
            if "episode" in info:
                self.episode_count += 1
                print(f"Episode {self.episode_count} completed")
                if self.episode_count >= self.max_episodes:
                    print(f"Reached maximum of {self.max_episodes} episodes. Stopping training.")
                    return False  # stops learning
        return True


device = "cuda" if torch.cuda.is_available() else "cpu"


def mask_fn(env):
    return env.get_action_mask()


hyperparameters = SLHyperParameters(
    training_epochs=8,
    learning_rate=0.00132,
    momentum=0.9,
    batch_size=32,
    optimizer_type="Adam",
)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
log_folder = os.path.join(project_root, "tensorboard_logs")
os.makedirs(log_folder, exist_ok=True)

print("TensorBoard logs will be saved to:", log_folder)
log_interval = 1  # log every n steps
num_existing_logs = len(
    [name for name in os.listdir(log_folder) if os.path.isdir(os.path.join(log_folder, name))]
)
run_name = f"NAS_run {num_existing_logs}"
logger = Logger(
    folder=log_folder, output_formats=[TensorBoardOutputFormat(os.path.join(log_folder, run_name))]
)
writer = SummaryWriter(log_dir=os.path.join(log_folder, run_name))
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
        policy_seed: Optional[int] = None,
        rl_learning_rate: float = 0.001,
        hyperparameters: SLHyperParameters = hyperparameters,
        reward_weights: Weights | None = None,
    ):
        self.env = CustomEnv(
            device=device,
            hyperparameters=hyperparameters,
            tb_logger=tb_logger,
            reward_strategy=WeightedSumRS(weights=reward_weights)
            if reward_weights
            else WeightedSumRS(Weights(accuracy=0.5, flops=0.5)),
        )
        self.model = policy_algorithm_class(
            policy=policy,
            env=self.env,
            verbose=1,
            gamma=1,  # type: ignore # extremely important to have gamma=1 for maximum discount
            device="cpu",
            learning_rate=rl_learning_rate,
            seed=policy_seed,
            n_steps=10,  # type: ignore
            normalize_advantage=False,  # type: ignore
        )
        self.model.set_logger(tb_logger.logger)
        self.model_save_path = f"{self.MODEL_SAVE_DIRECTORY}{self.model.__class__.__name__}"
        self.check_directories()

    def train(self, total_timesteps: int = 30000):
        self.model.learn(
            total_timesteps=total_timesteps, callback=EpisodeLimitCallback(max_episodes=5000)
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
            done = False
            episode_reward = 0

            while not done:
                if isinstance(obs, tuple):
                    obs = np.concatenate([np.array(o).flatten() for o in obs])

                obs_tensor = torch.as_tensor(obs).float().unsqueeze(0)
                actions, value, log_probs = self.model.policy.forward(obs_tensor)
                actions = actions.cpu().numpy().squeeze(0)
                obs, reward, terminated, truncated, _ = self.env.step(actions)
                done = terminated or truncated
                episode_reward += reward  # type: ignore

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
