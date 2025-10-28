from __future__ import annotations
import optuna
from typing import Dict, Any, Callable
from pydantic import BaseModel
from rich.console import Console
import torch
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from src.model_module.sb_three import SBThreeAgent
from src.classification_module.reward import Weights, Baselines


class HyperparameterSearchSpace(BaseModel):
    """Hyperparameter search space configuration"""

    # Architecture training hyperparameters
    training_epochs_min: int = 10
    training_epochs_max: int = 30

    arch_lr_min: float = 1e-4  # Architecture optimizer learning rate
    arch_lr_max: float = 1e-2

    arch_momentum_min: float = 0.5
    arch_momentum_max: float = 0.95

    batch_size_choices: list = [32, 64, 128, 256]

    # RL agent hyperparameters
    rl_lr_min: float = 1e-5
    rl_lr_max: float = 1e-2

    # Reward function weights
    accuracy_weight_min: float = 4.0
    accuracy_weight_max: float = 10.0

    f1_weight_min: float = 5.0
    f1_weight_max: float = 15.0

    test_loss_weight_min: float = 3.0
    test_loss_weight_max: float = 8.0

    flops_weight_min: float = 1.0
    flops_weight_max: float = 4.0

    runtime_weight_min: float = 1.0
    runtime_weight_max: float = 5.0


class HyperparameterOptimizer:
    """Bayesian optimization for hyperparameter tuning using Optuna

    Each trial uses its own independent environment copy to ensure proper isolation.
    """

    def __init__(
        self,
        search_space: HyperparameterSearchSpace = HyperparameterSearchSpace(),
        n_trials: int = 20,
        timeout: int | None = None,
    ):
        self.search_space = search_space
        self.n_trials = n_trials
        self.timeout = timeout
        self.console = Console()
        self.best_params: Dict[str, Any] | None = None
        self.best_value: float = 0.0

    def _objective(self, trial: optuna.Trial, agent_class, total_timesteps: int) -> float:
        """Objective function for Optuna optimization"""

        # Sample architecture training hyperparameters
        training_epochs = trial.suggest_int(
            "training_epochs",
            self.search_space.training_epochs_min,
            self.search_space.training_epochs_max,
        )

        arch_lr = trial.suggest_float(
            "arch_lr",
            self.search_space.arch_lr_min,
            self.search_space.arch_lr_max,
            log=True,  # Log scale for learning rates
        )

        arch_momentum = trial.suggest_float(
            "arch_momentum",
            self.search_space.arch_momentum_min,
            self.search_space.arch_momentum_max,
        )

        batch_size = trial.suggest_categorical(
            "batch_size",
            self.search_space.batch_size_choices,
        )

        # Sample RL agent hyperparameters
        rl_lr = trial.suggest_float(
            "rl_lr",
            self.search_space.rl_lr_min,
            self.search_space.rl_lr_max,
            log=True,
        )

        # Sample reward function weights
        accuracy_weight = trial.suggest_float(
            "accuracy_weight",
            self.search_space.accuracy_weight_min,
            self.search_space.accuracy_weight_max,
        )

        f1_weight = trial.suggest_float(
            "f1_weight",
            self.search_space.f1_weight_min,
            self.search_space.f1_weight_max,
        )

        test_loss_weight = trial.suggest_float(
            "test_loss_weight",
            self.search_space.test_loss_weight_min,
            self.search_space.test_loss_weight_max,
        )

        flops_weight = trial.suggest_float(
            "flops_weight",
            self.search_space.flops_weight_min,
            self.search_space.flops_weight_max,
        )

        runtime_weight = trial.suggest_float(
            "runtime_weight",
            self.search_space.runtime_weight_min,
            self.search_space.runtime_weight_max,
        )

        try:
            # Create agent with sampled hyperparameters
            agent = SBThreeAgent(
                policy_algorithm_class=agent_class,
                learning_rate=rl_lr,
                training_epochs=training_epochs,
                arch_learning_rate=arch_lr,
                arch_momentum=arch_momentum,
                batch_size=batch_size,
                reward_weights=Weights(
                    accuracy=accuracy_weight,
                    f1_score=f1_weight,
                    test_loss=test_loss_weight,
                    flops=flops_weight,
                    runtime=runtime_weight,
                ),
            )

            # Train agent
            agent.train(total_timesteps=total_timesteps)

            # Evaluate and return performance metric
            performance = agent.evaluate(num_episodes=5)

            return float(performance)

        except Exception as e:
            self.console.print(f"[bold red]Trial failed: {e}[/bold red]")
            return -10.0  # Return poor score on failure

    def optimize(
        self,
        agent_class: type[BaseAlgorithm],
        total_timesteps: int = 10000,
        study_name: str = "hyperparameter_optimization",
    ) -> Dict[str, Any]:
        """Run Bayesian optimization to find best hyperparameters"""

        self.console.print("[bold blue]Starting hyperparameter optimization...[/bold blue]")

        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
        )

        study.optimize(
            lambda trial: self._objective(trial, agent_class, total_timesteps),
            n_trials=self.n_trials,
            timeout=self.timeout,
        )

        self.best_params = study.best_params
        self.best_value = study.best_value

        self.console.print(f"[bold green]Optimization complete![/bold green]")
        self.console.print(f"[bold green]Best value: {self.best_value:.4f}[/bold green]")
        self.console.print(f"[bold cyan]Best parameters:[/bold cyan]")
        for param, value in self.best_params.items():
            self.console.print(f"  {param}: {value}")

        return self.best_params

    def get_best_params(self) -> Dict[str, Any]:
        """Get the best found hyperparameters"""
        if self.best_params is None:
            raise ValueError("Optimization has not been run yet")
        return self.best_params
