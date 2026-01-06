

from stable_baselines3 import PPO
from src.agent.action_masking.action_masking_policy import CustomMaskablePolicy
from src.environment.reward.reward import Weights
from src.environment.reward.weighted_sum import WeightedSumRS
from src.utils.architecture import Architecture
import src.agent.agent as agent
from src.utils.logger import get_layers_as_str
import time
from rich.console import Console

# Initialize the RL agent
rl_agent = agent.RLAgent(
    policy_algorithm_class=PPO,
    policy=CustomMaskablePolicy,
    policy_seed=42,
    reward_strategy=WeightedSumRS(Weights(accuracy=0.8, flops=0.2)),
)

rl_agent.load_model(PPO.__name__)

# Train best fifty architectures for x epochs
EPOCHS = 100
cache = agent.tb_logger.best_fifty_cache.cache
num_classes_train, num_classes_test = rl_agent.env.data_importer.get_num_classes()
dimensions = rl_agent.env.data_importer.get_dimensions()

console = Console()

rank: int = 1
for entry in cache:    
    print(get_layers_as_str(entry.architecture.layers, is_for_console=True))
    architecture = Architecture(entry.architecture, num_classes=num_classes_train, input_dimensions=dimensions)
    start_time = time.time()
    for epoch in range(EPOCHS):
        progress = (epoch + 1) / EPOCHS * 100
        with console.status(
            f"[bold blue]Training model entry with rank '{rank}', Progress {int(progress)}% ({time.time() - start_time:.2f}s)[/bold blue]"
        ):
            optimizer = rl_agent.env.hyperparameters.get_optimizer(architecture.parameters())
            rl_agent.env.trainer.train(architecture, optimizer)

    end_time = time.time()
    training_time = end_time - start_time
    rank += 1

    with console.status(f"[bold blue]Evaluating model entry with rank '{rank}'[/bold blue]"):
        metrics = rl_agent.env.evaluator.evaluate(architecture, training_time=training_time)

    console.print("Metrics for architecture with rank '{rank}':")
    console.print(metrics)

