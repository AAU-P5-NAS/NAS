import argparse

def ParseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimize-hyperparameters",
        type=int,
        default=None,
        help="If set, runs hyperparameter optimization with given number of trials",
    )
    parser.add_argument("--clean-saved-models", action="store_true")
    parser.add_argument("--report-exception", action="store_true")
    parser.add_argument(
        "--policy-seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--torch-seed", type=int, default=None, help="Random seed for classifier initialization"
    )
    parser.add_argument(
        "--optuna-seed", type=int, default=None, help="Random seed for hyperparameter optimization"
    )
    parser.add_argument("--load-model", type=str, default=None, help="Name of the model to load")

    args = parser.parse_args()
    return args