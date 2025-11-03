"""Multi-stage training system that alternates between architecture search and hyperparameter optimization"""

from __future__ import annotations
from typing import Dict, Any
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import A2C
from src.model_module.sb_three import SBThreeAgent
from src.model_module.hyperparameter_tuning import (
    SLHyperparameterOptimizer,
    RLHyperparameterOptimizer,
)
from src.model_module.hyperparameters import HyperparameterSearchSpace
from src.classification_module.reward import Weights
import json
import os


class StageConfig(BaseModel):
    """Configuration for a training stage"""

    name: str
    timesteps: int
    n_trials: int = 10  # For hyperparameter optimization stages


class MultiStageTrainer:
    """Coordinates multi-stage training alternating between architecture search and hyperparameter optimization"""

    def __init__(
        self,
        agent_class: type[BaseAlgorithm] = A2C,
        search_space: HyperparameterSearchSpace = HyperparameterSearchSpace(),
        output_dir: str = "saved_models",
    ):
        self.agent_class = agent_class
        self.search_space = search_space
        self.output_dir = output_dir
        self.console = Console()
        self.best_hyperparameters: Dict[str, Any] | None = None
        self.best_architecture_reward: float = -float("inf")
        self.stage_history: list[Dict[str, Any]] = []

    def stage_1_architecture_search(
        self,
        timesteps: int = 50000,
        hyperparams: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Stage 1: Find the best architecture using default or provided hyperparameters"""

        self.console.print("\n[bold cyan]╔════════════════════════════════════════╗[/bold cyan]")
        self.console.print("[bold cyan]║  Stage 1: Architecture Search          ║[/bold cyan]")
        self.console.print("[bold cyan]╚════════════════════════════════════════╝[/bold cyan]\n")

        # Use optimized hyperparameters if available
        if hyperparams is None:
            hyperparams = {
                "training_epochs": 15,
                "arch_lr": 0.001,
                "arch_momentum": 0.9,
                "batch_size": 64,
                "rl_lr": 0.001,
            }

        self.console.print(f"[yellow]Using hyperparameters:[/yellow] {hyperparams}")

        # Create agent with given hyperparameters
        agent = SBThreeAgent(
            policy_algorithm_class=self.agent_class,
            learning_rate=hyperparams["rl_lr"],
            training_epochs=hyperparams["training_epochs"],
            arch_learning_rate=hyperparams["arch_lr"],
            arch_momentum=hyperparams["arch_momentum"],
            batch_size=hyperparams["batch_size"],
        )

        # Train agent to explore architectures
        self.console.print(
            f"\n[bold green]Training agent for {timesteps} timesteps...[/bold green]"
        )
        agent.train(total_timesteps=timesteps)

        # Evaluate the agent
        avg_reward = agent.evaluate(num_episodes=10)

        # Save model
        model_path = os.path.join(self.output_dir, "stage1_best_architecture")
        agent.save_model(model_path)

        stage_info = {
            "stage": 1,
            "type": "architecture_search",
            "avg_reward": avg_reward,
            "hyperparameters": hyperparams,
            "model_path": model_path,
        }
        self.stage_history.append(stage_info)

        self.console.print("\n[bold green]✓ Stage 1 Complete[/bold green]")
        self.console.print(f"[bold green]Average reward: {avg_reward:.4f}[/bold green]")

        if avg_reward > self.best_architecture_reward:
            self.best_architecture_reward = avg_reward
            self.console.print("[bold yellow]New best architecture found![/bold yellow]")

        return stage_info

    def stage_2_hyperparameter_optimization(
        self,
        sl_trials: int = 20,
        rl_trials: int = 5,
        rl_timesteps: int = 10000,
        optimize_rl: bool = False,
    ) -> Dict[str, Any]:
        """Stage 2: Optimize SL hyperparameters (and optionally RL)"""

        self.console.print("\n[bold cyan]╔════════════════════════════════════════╗[/bold cyan]")
        self.console.print("[bold cyan]║  Stage 2: Hyperparameter Optimization  ║[/bold cyan]")
        self.console.print("[bold cyan]╚════════════════════════════════════════╝[/bold cyan]\n")

        self.console.print("[yellow]Step 1: Optimizing SL hyperparameters...[/yellow]")
        sl_optimizer = SLHyperparameterOptimizer(
            search_space=self.search_space,
            n_trials=sl_trials,
        )

        sl_best_params = sl_optimizer.optimize(
            study_name="multi_stage_sl_optimization",
        )

        rl_best_params = {}
        if optimize_rl:
            self.console.print(
                "\n[yellow]Step 2: Optimizing RL hyperparameters (discrete choices only)...[/yellow]"
            )
            self.console.print(
                "[bold yellow]Warning: This is computationally expensive![/bold yellow]"
            )

            rl_optimizer = RLHyperparameterOptimizer(
                search_space=self.search_space,
                n_trials=rl_trials,
            )

            rl_best_params = rl_optimizer.optimize(
                agent_class=self.agent_class,
                sl_hyperparams=sl_best_params,
                total_timesteps=rl_timesteps,
                study_name="multi_stage_rl_optimization",
            )
        else:
            self.console.print(
                "\n[yellow]Step 2: Skipping RL hyperparameter optimization (too expensive)[/yellow]"
            )
            self.console.print("[yellow]Using default RL hyperparameters[/yellow]")
            rl_best_params = {"rl_lr": 0.001}

        best_params = {**sl_best_params, **rl_best_params}

        params_file = os.path.join(self.output_dir, "stage2_best_hyperparameters.json")
        with open(params_file, "w") as f:
            json.dump(best_params, f, indent=4)

        self.best_hyperparameters = best_params

        stage_info = {
            "stage": 2,
            "type": "hyperparameter_optimization",
            "best_hyperparameters": best_params,
            "sl_params": sl_best_params,
            "rl_params": rl_best_params,
            "params_file": params_file,
        }
        self.stage_history.append(stage_info)

        self.console.print("\n[bold green]✓ Stage 2 Complete[/bold green]")
        self.console.print(f"[bold green]Best hyperparameters saved to {params_file}[/bold green]")

        return stage_info

    def stage_3_architecture_with_optimized_params(
        self,
        timesteps: int = 50000,
    ) -> Dict[str, Any]:
        """Stage 3: Architecture search with optimized hyperparameters"""

        self.console.print("\n[bold cyan]╔════════════════════════════════════════╗[/bold cyan]")
        self.console.print("[bold cyan]║  Stage 3: Architecture Search (Optimized) ║[/bold cyan]")
        self.console.print("[bold cyan]╚════════════════════════════════════════╝[/bold cyan]\n")

        if self.best_hyperparameters is None:
            self.console.print(
                "[bold red]No optimized hyperparameters found! Running Stage 2 first...[/bold red]"
            )
            self.stage_2_hyperparameter_optimization()

        self.console.print(
            f"[yellow]Using optimized hyperparameters:[/yellow] {self.best_hyperparameters}"
        )

        # Create agent with optimized hyperparameters
        agent = SBThreeAgent(
            policy_algorithm_class=self.agent_class,
            learning_rate=self.best_hyperparameters["rl_lr"],
            training_epochs=self.best_hyperparameters["training_epochs"],
            arch_learning_rate=self.best_hyperparameters["arch_lr"],
            arch_momentum=self.best_hyperparameters["arch_momentum"],
            batch_size=self.best_hyperparameters["batch_size"],
            reward_weights=Weights(
                accuracy=self.best_hyperparameters.get("accuracy_weight", 6.0),
                f1_score=self.best_hyperparameters.get("f1_weight", 10.0),
                test_loss=self.best_hyperparameters.get("test_loss_weight", 5.0),
                flops=self.best_hyperparameters.get("flops_weight", 2.0),
                runtime=self.best_hyperparameters.get("runtime_weight", 3.0),
            ),
        )

        # Train agent with optimized parameters
        self.console.print(
            f"\n[bold green]Training with optimized hyperparameters for {timesteps} timesteps...[/bold green]"
        )
        agent.train(total_timesteps=timesteps)

        # Evaluate
        avg_reward = agent.evaluate(num_episodes=10)

        # Save model
        model_path = os.path.join(self.output_dir, "stage3_final_architecture")
        agent.save_model(model_path)

        stage_info = {
            "stage": 3,
            "type": "architecture_search_optimized",
            "avg_reward": avg_reward,
            "hyperparameters": self.best_hyperparameters,
            "model_path": model_path,
        }
        self.stage_history.append(stage_info)

        self.console.print("\n[bold green]✓ Stage 3 Complete[/bold green]")
        self.console.print(f"[bold green]Average reward: {avg_reward:.4f}[/bold green]")

        if avg_reward > self.best_architecture_reward:
            self.best_architecture_reward = avg_reward
            self.console.print("[bold yellow]New best architecture found![/bold yellow]")

        return stage_info

    def print_summary(self):
        """Print summary of all stages"""

        table = Table(title="Multi-Stage Training Summary")
        table.add_column("Stage", style="cyan", no_wrap=True)
        table.add_column("Type", style="magenta")
        table.add_column("Average Reward", style="green")

        for stage_info in self.stage_history:
            stage_num = stage_info.get("stage", "?")
            stage_type = stage_info.get("type", "unknown")
            reward = stage_info.get("avg_reward", "N/A")

            if isinstance(reward, float):
                reward = f"{reward:.4f}"

            table.add_row(str(stage_num), stage_type, reward)

        self.console.print("\n")
        self.console.print(table)

        self.console.print(
            f"\n[bold green]Best Architecture Reward: {self.best_architecture_reward:.4f}[/bold green]"
        )

    def run_all_stages(
        self,
        stage1_timesteps: int = 50000,
        stage2_timesteps: int = 10000,
        stage2_trials: int = 10,
        stage3_timesteps: int = 50000,
        optimize_rl: bool = False,
        max_iterations: int = 5,
        improvement_threshold: float = 0.001,
        no_improvement_limit: int = 2,
    ):
        """Run iterative multi-stage training until convergence"""

        self.console.print(
            "[bold blue]╔══════════════════════════════════════════════════╗[/bold blue]"
        )
        self.console.print(
            "[bold blue]║  Iterative Multi-Stage Training Pipeline        ║[/bold blue]"
        )
        self.console.print(
            "[bold blue]║  1. Architecture Search                        ║[/bold blue]"
        )
        self.console.print(
            "[bold blue]║  2. Hyperparameter Optimization                ║[/bold blue]"
        )
        self.console.print(
            "[bold blue]║  → Continues until convergence or max iterations ║[/bold blue]"
        )
        self.console.print(
            "[bold blue]╚══════════════════════════════════════════════════╝[/bold blue]\n"
        )

        stage1_results = self.stage_1_architecture_search(timesteps=stage1_timesteps)
        last_arch_reward = self.best_architecture_reward
        iterations_without_improvement = 0

        stage2_results = self.stage_2_hyperparameter_optimization(
            sl_trials=stage2_trials,
            rl_trials=5,
            rl_timesteps=stage2_timesteps,
            optimize_rl=optimize_rl,
        )

        iteration = 1
        while iteration <= max_iterations:
            self.console.print(
                "\n[bold cyan]═══════════════════════════════════════════════════[/bold cyan]"
            )
            self.console.print(
                f"[bold cyan]Iteration {iteration}/{max_iterations} - Searching with optimized parameters[/bold cyan]"
            )
            self.console.print(
                "[bold cyan]═══════════════════════════════════════════════════[/bold cyan]\n"
            )

            self.stage_3_architecture_with_optimized_params(
                timesteps=stage3_timesteps,
            )

            current_reward = self.best_architecture_reward
            improvement = current_reward - last_arch_reward

            if improvement > improvement_threshold:
                self.console.print(
                    f"[bold green]✓ Improvement of {improvement:.4f} detected![/bold green]"
                )
                iterations_without_improvement = 0
                last_arch_reward = current_reward

                if iteration < max_iterations:
                    self.console.print("\n[yellow]Re-optimizing hyperparameters...[/yellow]")
                    self.stage_2_hyperparameter_optimization(
                        sl_trials=stage2_trials,
                        rl_trials=5,
                        rl_timesteps=stage2_timesteps,
                        optimize_rl=False,
                    )
            else:
                iterations_without_improvement += 1
                self.console.print(
                    f"[yellow]No significant improvement ({improvement:.4f} < {improvement_threshold})[/yellow]"
                )
                self.console.print(
                    f"[yellow]Iterations without improvement: {iterations_without_improvement}/{no_improvement_limit}[/yellow]"
                )

                if iterations_without_improvement >= no_improvement_limit:
                    self.console.print(
                        "\n[bold red]Stopping: No improvement detected for too many iterations[/bold red]"
                    )
                    break

            if iteration < max_iterations:
                self.stage_2_hyperparameter_optimization(
                    sl_trials=stage2_trials,
                    rl_trials=5,
                    rl_timesteps=stage2_timesteps,
                    optimize_rl=False,
                )

            iteration += 1

        self.print_summary()

        return {
            "stage1": stage1_results,
            "stage2": stage2_results,
            "best_reward": self.best_architecture_reward,
            "total_iterations": iteration,
            "converged": iterations_without_improvement >= no_improvement_limit,
        }
