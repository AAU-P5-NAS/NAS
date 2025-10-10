from src.model_module.environment import CustomEnv
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy


class SBThreeAgent:
    def __init__(
        self,
        policy_algorithm_class: type[BaseAlgorithm],
        policy: type[BasePolicy] = ActorCriticPolicy,
        learning_rate: float = 0.005,
    ):
        self.env: CustomEnv = CustomEnv()
        self.model = policy_algorithm_class(
            policy=policy, env=self.env, verbose=1, learning_rate=learning_rate
        )

    def train(self, total_timesteps: int = 10000):
        self.model.learn(total_timesteps=total_timesteps)
