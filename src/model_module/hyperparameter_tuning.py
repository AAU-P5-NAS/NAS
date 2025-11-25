"""Hyperparameter optimization with separated RL and SL tuning"""
""" 
from __future__ import annotations
import optuna
from typing import Dict, Any, Optional
from rich.console import Console
import torch
from stable_baselines3.common.base_class import BaseAlgorithm

from data_module.dataset import DatasetOption
from src.model_module.hyperparameters import HyperparameterSearchSpace
from src.classification_module.reward import Weights, WeightedSumRS
from src.classification_module.train import Trainer
from src.data_module.importer import DataImporter
from src.utils.network_utils import (
    NetworkConfig,
    LayerConfig,
    LayerType,
    OutChannels,
    KernelSize,
    Stride,
    ActivationFunction,
    PoolMode,
    LinearUnits,
)
from torch.nn import CrossEntropyLoss


def create_standard_architecture() -> NetworkConfig:
    layers = [
        LayerConfig(
            layer_type=LayerType.CONV,
            out_channels=OutChannels.CH_32,
            kernel_size=KernelSize.KS_3,
            stride=Stride.S_1,
            activation=ActivationFunction.RELU,
        ),
        LayerConfig(
            layer_type=LayerType.POOL,
            kernel_size=KernelSize.KS_3,
            stride=Stride.S_2,
            pool_mode=PoolMode.MAX,
            activation=ActivationFunction.NONE,
        ),
        LayerConfig(
            layer_type=LayerType.CONV,
            out_channels=OutChannels.CH_64,
            kernel_size=KernelSize.KS_3,
            stride=Stride.S_1,
            activation=ActivationFunction.RELU,
        ),
        LayerConfig(
            layer_type=LayerType.POOL,
            kernel_size=KernelSize.KS_3,
            stride=Stride.S_2,
            pool_mode=PoolMode.MAX,
            activation=ActivationFunction.NONE,
        ),
        LayerConfig(
            layer_type=LayerType.LINEAR,
            linear_units=LinearUnits.LU_128,
            activation=ActivationFunction.RELU,
        ),
        LayerConfig(
            layer_type=LayerType.LINEAR,
            linear_units=LinearUnits.LU_256,
            activation=ActivationFunction.NONE,
        ),
    ]

    return NetworkConfig(layers=layers)


class SLHyperparameterOptimizer:

    def __init__(
        self,
        search_space: HyperparameterSearchSpace,
        n_trials: int = 20,
        timeout: Optional[int] = None,
    ):
        self.search_space = search_space
        self.n_trials = n_trials
        self.timeout = timeout
        self.console = Console()
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_value: float = 0.0
        self.standard_architecture = create_standard_architecture()

    def _objective(self, trial: optuna.Trial) -> float:

        sl = self.search_space.sl_hyperparameters

        learning_rate = trial.suggest_float(
            "arch_lr",
            sl.learning_rate_min,
            sl.learning_rate_max,
            log=True,
        )

        momentum = trial.suggest_float(
            "arch_momentum",
            sl.momentum_min,
            sl.momentum_max,
        )

        batch_size = trial.suggest_categorical(
            "batch_size",
            sl.batch_size_choices,
        )

        training_epochs = trial.suggest_categorical(
            "training_epochs",
            sl.training_epochs_choices,
        )

        optimizer_type = trial.suggest_categorical(
            "optimizer_type",
            ["SGD", "Adam", "RMSprop"],
        )

        reward_weights_config = self.search_space.reward_weights
        accuracy_weight = trial.suggest_float(
            "accuracy_weight",
            reward_weights_config.accuracy_weight_min,
            reward_weights_config.accuracy_weight_max,
        )
        f1_weight = trial.suggest_float(
            "f1_weight",
            reward_weights_config.f1_weight_min,
            reward_weights_config.f1_weight_max,
        )
        test_loss_weight = trial.suggest_float(
            "test_loss_weight",
            reward_weights_config.test_loss_weight_min,
            reward_weights_config.test_loss_weight_max,
        )
        flops_weight = trial.suggest_float(
            "flops_weight",
            reward_weights_config.flops_weight_min,
            reward_weights_config.flops_weight_max,
        )
        runtime_weight = trial.suggest_float(
            "runtime_weight",
            reward_weights_config.runtime_weight_min,
            reward_weights_config.runtime_weight_max,
        )

        try:
            data_importer = DataImporter(
                max_per_class=1000, dataset_option=DatasetOption.EMNIST_BALANCED
            )
            train_loader, test_loader = data_importer.get_dataloaders(batch_size=batch_size)
            train_num_classes, test_num_classes = data_importer.get_num_classes()
            cnn_builder = CNNBuilder(
                rl_config=self.standard_architecture, num_classes=train_num_classes
            )
            model = cnn_builder.build()

            if optimizer_type == "SGD":
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=learning_rate,
                    momentum=momentum,
                )
            elif optimizer_type == "Adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            elif optimizer_type == "RMSprop":
                optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
            else:
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=learning_rate,
                    momentum=momentum,
                )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            trainer = Trainer(
                dataloaders=(train_loader, test_loader),
                model=model.to(device),
                loss_function=CrossEntropyLoss().to(device),
                optimizer=optimizer,
                num_classes=train_num_classes,
            )

            for epoch in range(training_epochs):
                trainer.train()

            metrics = trainer.test()

            reward_weights = Weights(
                accuracy=accuracy_weight,
                f1_score=f1_weight,
                test_loss=test_loss_weight,
                flops=flops_weight,
                runtime=runtime_weight,
            )
            reward_calculator = WeightedSumRS(weights=reward_weights)
            reward = reward_calculator.compute_reward(metrics)

            return float(reward)

        except Exception as e:
            self.console.print(f"[bold red]Trial failed: {e}[/bold red]")
            return -10.0

    def optimize(
        self,
        study_name: str = "sl_hyperparameter_optimization",
    ) -> Dict[str, Any]:

        self.console.print(
            "[bold blue]Starting SL hyperparameter optimization with standard architecture...[/bold blue]"
        )

        study = optuna.create_study(direction="maximize", study_name=study_name)

        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
        )

        self.best_params = study.best_params
        self.best_value = study.best_value

        self.console.print("[bold green]Optimization complete![/bold green]")
        self.console.print(f"[bold green]Best value: {self.best_value:.4f}[/bold green]")
        self.console.print("[bold cyan]Best SL parameters:[/bold cyan]")
        for param, value in self.best_params.items():
            self.console.print(f"  {param}: {value}")

        return self.best_params


class RLHyperparameterOptimizer:

    def __init__(
        self,
        search_space: HyperparameterSearchSpace,
        n_trials: int = 10,
        timeout: Optional[int] = None,
    ):
        self.search_space = search_space
        self.n_trials = n_trials
        self.timeout = timeout
        self.console = Console()
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_value: float = 0.0

    def _objective(
        self,
        trial: optuna.Trial,
        agent_class: type[BaseAlgorithm],
        total_timesteps: int,
        sl_hyperparams: Dict[str, Any],
    ) -> float:

        learning_rate_choice = trial.suggest_categorical(
            "rl_lr",
            [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
        )

        try:
            from model_module.agent import SBThreeAgent

            agent = SBThreeAgent(
                policy_algorithm_class=agent_class,
                learning_rate=learning_rate_choice,
                training_epochs=sl_hyperparams.get("training_epochs", 15),
                arch_learning_rate=sl_hyperparams.get("arch_lr", 0.001),
                arch_momentum=sl_hyperparams.get("arch_momentum", 0.9),
                batch_size=sl_hyperparams.get("batch_size", 64),
            )

            agent.train(total_timesteps=total_timesteps)
            performance = agent.evaluate(num_episodes=5)

            return float(performance)

        except Exception as e:
            self.console.print(f"[bold red]Trial failed: {e}[/bold red]")
            return -10.0

    def optimize(
        self,
        agent_class: type[BaseAlgorithm],
        sl_hyperparams: Dict[str, Any],
        total_timesteps: int = 10000,
        study_name: str = "rl_hyperparameter_optimization",
    ) -> Dict[str, Any]:

        self.console.print(
            "[bold blue]Starting RL hyperparameter optimization (discrete choices only)...[/bold blue]"
        )
        self.console.print(
            "[yellow]Warning: RL hyperparameter tuning is computationally expensive![/yellow]"
        )

        study = optuna.create_study(direction="maximize", study_name=study_name)

        study.optimize(
            lambda trial: self._objective(trial, agent_class, total_timesteps, sl_hyperparams),
            n_trials=self.n_trials,
            timeout=self.timeout,
        )

        self.best_params = study.best_params
        self.best_value = study.best_value

        self.console.print("[bold green]Optimization complete![/bold green]")
        self.console.print(f"[bold green]Best value: {self.best_value:.4f}[/bold green]")
        self.console.print("[bold cyan]Best RL parameters:[/bold cyan]")
        for param, value in self.best_params.items():
            self.console.print(f"  {param}: {value}")

        return self.best_params


class HyperparameterOptimizer:

    def __init__(self, *args, **kwargs):
        self.console = Console()
        self.console.print(
            "[bold yellow]Warning: HyperparameterOptimizer is deprecated.[/bold yellow]"
        )
        self.console.print(
            "[yellow]Use SLHyperparameterOptimizer and RLHyperparameterOptimizer separately.[/yellow]"
        )

    def optimize(self, *args, **kwargs):
        raise NotImplementedError(
            "Use SLHyperparameterOptimizer and RLHyperparameterOptimizer separately"
        )
 """