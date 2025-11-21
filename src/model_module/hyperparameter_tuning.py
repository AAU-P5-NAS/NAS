"""Hyperparameter optimization with separated RL and SL tuning"""

from __future__ import annotations
import traceback
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
from src.utils.cnn_builder import CNNBuilder
from torch.nn import CrossEntropyLoss

console = Console()


def create_standard_architecture(number_of_classes: int, dropout_rate_linear_layer: float) -> torch.nn.Sequential:
    """Create a standard architecture for SL hyperparameter tuning"""
    model = torch.nn.Sequential(
        # Block 1
        torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        # Block 2
        torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(),
        torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        # Block 3
        torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(256),
        torch.nn.ReLU(),
        torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(256),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        # Fully connected
        torch.nn.Flatten(),
        torch.nn.Linear(256 * 4 * 4, 512),  # Adjusted input size after removing Block 4
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout_rate_linear_layer),
        torch.nn.Linear(512, number_of_classes),
    )

    for layer in model:
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)  # Xavier/Glorot uniform initialization
            if layer.bias is not None:
                layer.bias.data.fill_(0.0)

    return model


class SLHyperparameterOptimizer:
    """Optimize Supervised Learning hyperparameters using a standard architecture"""

    trials_run = 0
    TRAINING_EPOCHS = 25

    def __init__(
        self,
        search_space: HyperparameterSearchSpace,
        n_trials: int = 20,
        timeout: Optional[int] = None,
        dataset_option: DatasetOption = DatasetOption.CIFAR_10,
        seed: Optional[int] = None,
    ):
        self.search_space = search_space
        self.n_trials = n_trials
        self.timeout = timeout
        self.console = Console()
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_value: float = 0.0
        self.standard_architecture: torch.nn.Sequential | None = None
        self.dataset_option = dataset_option
        self.seed = seed

    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for SL hyperparameter optimization"""
        with console.status(
            f"Optimizing hyperparameters (Trial {self.trials_run+1}/{self.n_trials})..."
        ):
            sl = self.search_space.sl_hyperparameters

            with console.status("Suggesting hyperparameters (learning rate)..."):
                learning_rate = trial.suggest_float(
                    "arch_lr",
                    sl.learning_rate_min,
                    sl.learning_rate_max,
                    log=True,
                )

            with console.status("Suggesting hyperparameters (batch size)..."):
                batch_size = trial.suggest_categorical(
                    "batch_size",
                    sl.batch_size_choices,
                )

            with console.status("Suggesting hyperparameters (optimizer type)..."):
                optimizer_type = trial.suggest_categorical(
                    "optimizer_type",
                    ["SGD", "Adam", "RMSprop"],
                )

            with console.status("Suggesting hyperparameters (dropout rates)..."):
                """dropout_rate_pooling_layer = trial.suggest_float(
                    "dropout_rate_pooling_layer",
                    sl.dropout_rate_pooling_layer_min,
                    sl.dropout_rate_pooling_layer_max,
                )"""
                dropout_rate_linear_layer = trial.suggest_float(
                    "dropout_rate_linear_layer",
                    sl.dropout_rate_linear_layer_min,
                    sl.dropout_rate_linear_layer_max,
                )

            try:
                with console.status("Initializing data importer..."):
                    data_importer = DataImporter(dataset_option=self.dataset_option)
                    train_loader, test_loader = data_importer.get_dataloaders(batch_size=batch_size)
                    train_num_classes, test_num_classes = data_importer.get_num_classes()
                with console.status("Creating standard architecture..."):
                    self.standard_architecture = create_standard_architecture(train_num_classes, dropout_rate_linear_layer)
                if optimizer_type == "SGD":
                    with console.status("Suggesting hyperparameters (momentum)..."):
                        momentum = trial.suggest_float(
                            "arch_momentum",
                            sl.momentum_min,
                            sl.momentum_max,
                        )
                    
                    optimizer = torch.optim.SGD(
                        self.standard_architecture.parameters(),
                        lr=learning_rate,
                        momentum=momentum,
                    )

                elif optimizer_type == "Adam":
                    optimizer = torch.optim.Adam(
                        self.standard_architecture.parameters(), lr=learning_rate
                    )

                elif optimizer_type == "RMSprop":
                    optimizer = torch.optim.RMSprop(
                        self.standard_architecture.parameters(), lr=learning_rate
                    )
                    
                else:
                    raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

                with console.status("Initializing trainer..."):
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    trainer = Trainer(
                        dataloaders=(train_loader, test_loader),
                        model=self.standard_architecture.to(device),
                        loss_function=CrossEntropyLoss().to(device),
                        optimizer=optimizer,
                        num_classes=train_num_classes,
                        dimensions=data_importer.get_dimensions(),
                    )

                for epoch in range(self.TRAINING_EPOCHS):
                    with console.status(f"Training model (epoch {epoch + 1}/{self.TRAINING_EPOCHS})..."):
                        trainer.train()

                metrics = trainer.test()
            
                reward_calculator = WeightedSumRS(weights=Weights(accuracy=1.0))
                reward = reward_calculator.compute_reward(metrics)

            except Exception as e:
                self.console.print(f"[bold red]Trial failed: {e}[/bold red]")
                tb = traceback.extract_tb(e.__traceback__)
                for frame in tb:
                    print(f"File: {frame.filename}, Line: {frame.lineno}, Function: {frame.name}")
                print(f"Error: {e}")
                return -9999999
            
            finally:
                self.trials_run += 1

            return float(reward)


    def optimize(
        self,
        study_name: str = "sl_hyperparameter_optimization",
    ) -> Dict[str, Any]:
        """Run Bayesian optimization for SL hyperparameters"""

        self.console.print(
            "[bold blue]Starting SL hyperparameter optimization with standard architecture...[/bold blue]"
        )

        with console.status("Creating Study..."):
            sampler = optuna.samplers.NSGAIIISampler(seed=self.seed)
            study = optuna.create_study(direction="maximize", study_name=study_name, sampler=sampler)

        
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
        )

        with console.status("Writing results..."):
            self.best_params = study.best_params
            self.best_value = study.best_value

            self.console.print("[bold green]Optimization complete![/bold green]")
            self.console.print(f"[bold green]Best value: {self.best_value:.4f}[/bold green]")
            self.console.print("[bold cyan]Best SL parameters:[/bold cyan]")

            with open("sl_hyperparameter_optimization_results.txt", "w") as f:
                f.write("Best SL parameters:\n")
                for param, value in self.best_params.items():
                    f.write(f"{param}: {value}\n")
                f.write("\n")
                f.write(f"Best value: {self.best_value:.4f}\n")

            for param, value in self.best_params.items():
                self.console.print(f"  {param}: {value}")

        return self.best_params


class RLHyperparameterOptimizer:
    """Optimize RL hyperparameters using discrete choices only"""

    def __init__(
        self,
        search_space: HyperparameterSearchSpace,
        n_trials: int = 10,
        timeout: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.search_space = search_space
        self.n_trials = n_trials
        self.timeout = timeout
        self.seed = seed
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
        """Objective function for RL hyperparameter optimization"""

        learning_rate_choice = trial.suggest_categorical(
            "rl_lr",
            [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
        )

        try:
            from src.model_module.sb_three import SBThreeAgent
            
            """THIS DOES NOT SUPPORT REPRODUCIBILITY YET"""
            agent = SBThreeAgent(
                policy_algorithm_class=agent_class,
                rl_learning_rate=learning_rate_choice,
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
        """Run discrete optimization for RL hyperparameters"""

        self.console.print(
            "[bold blue]Starting RL hyperparameter optimization (discrete choices only)...[/bold blue]"
        )
        self.console.print(
            "[yellow]Warning: RL hyperparameter tuning is computationally expensive![/yellow]"
        )

        sampler = optuna.samplers.NSGAIIISampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", study_name=study_name, sampler=sampler)

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
    """Legacy optimizer"""

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
