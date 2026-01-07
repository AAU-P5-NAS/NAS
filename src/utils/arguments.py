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
        "--policy-seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--torch-seed", type=int, default=42, help="Random seed for classifier initialization"
    )
    parser.add_argument(
        "--optuna-seed", type=int, default=42, help="Random seed for hyperparameter optimization"
    )
    parser.add_argument("--load-model", type=str, default=None, help="Name of the model to load")
    parser.add_argument(
        "--evaluate-archive",
        action="store_true",
        help="If set, evaluates the given archive instead of training a new model.",
    )
    parser.add_argument(
        "--use-tchebycheff",
        action="store_true",
        help="If set, evaluates using Tchebycheff reward strategy.",
    )
    parser.add_argument(
        "--use-dominance-novelty",
        action="store_true",
        help="If set, evaluates using Dominance Novelty reward strategy.",
    )
    parser.add_argument(
        "--use-weighted-sum",
        action="store_true",
        help="If set, evaluates using Weighted Sum reward strategy.",
    )
    parser.add_argument(
        "--use-real-tchebycheff",
        action="store_true",
        help="If set, evaluates using Real Tchebycheff reward strategy.",
    )
    parser.add_argument(
        "--use-real-dominance-novelty",
        action="store_true",
        help="If set, evaluates using Real Dominance Novelty reward strategy.",
    )

    args = parser.parse_args()
    return args
