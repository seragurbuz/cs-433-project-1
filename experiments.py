"""Cross-validation experiments for the project models."""

from pathlib import Path

import numpy as np

from helpers import load_cleaned_data
from implementations import (
    least_squares,
    logistic_regression,
    mean_squared_error_gd,
    mean_squared_error_sgd,
    reg_logistic_regression,
    ridge_regression,
)
from metrics import f1_score, precision_score, recall_score


DATA_PATH = Path(__file__).parent / "dataset"
# These files contain features that were cleaned by dataloader.py
CLEANED_TRAIN_PATH = DATA_PATH / "cleaned_data" / "x_train_replaced.csv"
CLEANED_TEST_PATH = DATA_PATH / "cleaned_data" / "x_test_replaced.csv"
LABELS_PATH = DATA_PATH / "y_train.csv"
TEST_IDS_PATH = DATA_PATH / "x_test.csv"

# Shared cross-validation settings
N_FOLDS = 5
SPLIT_SEED = 42
NUM_EPOCHS = 1000
GAMMA = 0.1


# Each entry describes one model family and its parameter combinations.
# "closed_form" models solve for weights once; the other kinds use updates.
# Add or remove parameter dictionaries here to control the experiments.
MODEL_EXPERIMENTS = [
    {
        "name": "least_squares",
        "function": least_squares,
        "kind": "closed_form",
        "parameters": [{}],
    },
    {
        "name": "ridge_regression",
        "function": ridge_regression,
        "kind": "closed_form",
        "parameters": [
            {"lambda_": 0.0},
            {"lambda_": 1e-3},
            {"lambda_": 1e-2},
            {"lambda_": 1e-1},
        ],
    },
    {
        "name": "mean_squared_error_gd",
        "function": mean_squared_error_gd,
        "kind": "iterative",
        "parameters": [
            {"num_updates": 1000, "gamma": 0.01},
            {"num_updates": 1000, "gamma": 0.03},
            {"num_updates": 1000, "gamma": 0.1},
        ],
    },
    {
        "name": "mean_squared_error_sgd",
        "function": mean_squared_error_sgd,
        "kind": "iterative",
        "parameters": [
            {"num_updates": 10000, "gamma": 1e-4},
            {"num_updates": 10000, "gamma": 1e-3},
            {"num_updates": 10000, "gamma": 1e-2},
        ],
    },
    {
        "name": "logistic_regression",
        "function": logistic_regression,
        "kind": "logistic",
        "parameters": [
            {"num_updates": 1000, "gamma": 0.03},
            {"num_updates": 1000, "gamma": 0.1},
            {"num_updates": 3000, "gamma": 0.1},
        ],
    },
    {
        "name": "reg_logistic_regression",
        "function": reg_logistic_regression,
        "kind": "regularized_logistic",
        "parameters": [
            {"lambda_": 1e-3, "num_updates": 1000, "gamma": 0.1},
            {"lambda_": 1e-2, "num_updates": 1000, "gamma": 0.1},
            {"lambda_": 1e-1, "num_updates": 1000, "gamma": 0.1},
        ],
    },
]


def _stratified_folds(y, n_folds, seed):
    """Return validation indices for reproducible stratified K-fold CV."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if n_folds > np.min(np.bincount((y == 1).astype(int))):
        raise ValueError("n_folds cannot exceed the size of the smallest class")

    rng = np.random.default_rng(seed)
    fold_indices = [[] for _ in range(n_folds)]
    for label in np.unique(y):
        # Split each class separately so every validation fold has both labels.
        indices = np.flatnonzero(y == label)
        indices = rng.permutation(indices)
        for fold_number, fold_part in enumerate(np.array_split(indices, n_folds)):
            fold_indices[fold_number].extend(fold_part.tolist())

    return [rng.permutation(indices) for indices in fold_indices]


def _standardize(train_x, validation_x):
    """Standardize validation features with training-fold statistics."""
    # Validation data must not influence the scaling statistics.
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0] = 1
    return (train_x - mean) / std, (validation_x - mean) / std


def _design_matrix(x):
    """Add an intercept column to standardized features."""
    return np.column_stack((np.ones(x.shape[0]), x))


def _fit_model(x_train, y_train, model_config, parameters, random_seed):
    """Fit one configured model and return its weights."""
    tx_train = _design_matrix(x_train)
    weights = np.zeros(tx_train.shape[1])
    function = model_config["function"]
    kind = model_config["kind"]
    if kind == "closed_form":
        # Closed-form methods do not use an initial weight vector or updates.
        if function is least_squares:
            weights, _ = function(y_train, tx_train)
        else:
            weights, _ = function(y_train, tx_train, parameters["lambda_"])
        return weights

    if kind in ("logistic", "regularized_logistic"):
        # Logistic implementations expect labels in {0, 1}; linear methods use {-1, 1}.
        fit_y = (y_train + 1) / 2
    else:
        fit_y = y_train

    # This makes SGD results reproducible for each fold.
    np.random.seed(random_seed)
    if kind == "regularized_logistic":
        weights, _ = function(
            fit_y,
            tx_train,
            parameters["lambda_"],
            weights,
            parameters["num_updates"],
            parameters["gamma"],
        )
    else:
        weights, _ = function(
            fit_y,
            tx_train,
            weights,
            parameters["num_updates"],
            parameters["gamma"],
        )
    return weights


def _threshold_candidates(scores):
    """Return score thresholds covering the full predicted-positive range."""
    # Percentiles provide a model-specific threshold grid without assuming
    # that different models produce scores on the same numerical scale.
    percentiles = np.percentile(scores, np.arange(0, 101))
    return np.unique(np.concatenate(([-np.inf, 0.0, np.inf], percentiles)))


def _best_threshold(validation_results):
    """Find the threshold with the highest mean F1 across validation folds."""
    # The same threshold is evaluated independently on every fold. This avoids
    # choosing a different decision rule for each validation fold.
    all_scores = np.concatenate(
        [validation_scores for validation_scores, _ in validation_results]
    )
    candidates = _threshold_candidates(all_scores)
    results = []
    for threshold in candidates:
        fold_f1_scores = [
            f1_score(y_true, np.where(scores >= threshold, 1, -1))
            for scores, y_true in validation_results
        ]
        results.append((float(np.mean(fold_f1_scores)), threshold, fold_f1_scores))
    # The first item is mean F1; on a tie, prefer the lower threshold.
    return max(results, key=lambda result: (result[0], -result[1]))


def run_model_experiment(
    x, y, model_config, parameters, n_folds=N_FOLDS, seed=SPLIT_SEED
):
    """Run CV for one model configuration and select its F1 threshold."""
    validation_folds = _stratified_folds(y, n_folds, seed)
    out_of_fold_scores = np.empty(len(y))
    validation_results = []
    zero_threshold_f1_scores = []

    all_indices = np.arange(len(y))
    for fold_number, validation_indices in enumerate(validation_folds, start=1):
        # Train on the other folds and score the held-out fold.
        training_indices = np.setdiff1d(all_indices, validation_indices, assume_unique=False)
        x_train, x_validation = _standardize(
            x[training_indices], x[validation_indices]
        )
        weights = _fit_model(
            x_train,
            y[training_indices],
            model_config,
            parameters,
            random_seed=seed + fold_number,
        )
        validation_scores = _design_matrix(x_validation) @ weights
        # Each row receives a score from a model that did not train on that row.
        out_of_fold_scores[validation_indices] = validation_scores
        validation_results.append((validation_scores, y[validation_indices]))

        # Zero is a useful baseline before tuning the decision threshold.
        zero_predictions = np.where(validation_scores >= 0, 1, -1)
        fold_f1 = f1_score(y[validation_indices], zero_predictions)
        zero_threshold_f1_scores.append(fold_f1)
        print(
            f"Fold {fold_number}/{n_folds}: "
            f"zero-threshold F1={fold_f1:.4f}"
        )

    best_f1, best_threshold, selected_fold_f1_scores = _best_threshold(
        validation_results
    )
    zero_predictions = np.where(out_of_fold_scores >= 0, 1, -1)
    best_predictions = np.where(
        out_of_fold_scores >= best_threshold, 1, -1
    )
    return {
        "name": model_config["name"],
        "parameters": parameters,
        "threshold": best_threshold,
        "mean_f1": best_f1,
        "fold_f1": selected_fold_f1_scores,
        "mean_zero_f1": float(np.mean(zero_threshold_f1_scores)),
        "oof_f1": f1_score(y, best_predictions),
        "precision": precision_score(y, best_predictions),
        "recall": recall_score(y, best_predictions),
    }


def run_threshold_experiment(
    x, y, n_folds=N_FOLDS, seed=SPLIT_SEED, num_epochs=NUM_EPOCHS, gamma=GAMMA
):
    """Keep the original logistic-only entry point for compatibility."""
    logistic_config = next(
        config
        for config in MODEL_EXPERIMENTS
        if config["name"] == "logistic_regression"
    )
    parameters = {"num_updates": num_epochs, "gamma": gamma}
    return run_model_experiment(
        x, y, logistic_config, parameters, n_folds=n_folds, seed=seed
    )


def main():
    """Load the cleaned data and run all configured model experiments."""
    x_train, _, y_train, _, _ = load_cleaned_data(
        CLEANED_TRAIN_PATH,
        CLEANED_TEST_PATH,
        LABELS_PATH,
        TEST_IDS_PATH,
    )
    # Run every model/parameter combination using the same CV procedure.
    results = []
    for model_config in MODEL_EXPERIMENTS:
        print(f"\n=== {model_config['name']} ===")
        for parameters in model_config["parameters"]:
            print(f"Parameters: {parameters}")
            result = run_model_experiment(
                x_train,
                y_train,
                model_config,
                parameters,
            )
            results.append(result)
            print(
                f"Mean CV F1={result['mean_f1']:.4f}, "
                f"threshold={result['threshold']:.6f}, "
                f"precision={result['precision']:.4f}, "
                f"recall={result['recall']:.4f}"
            )

    # Compare configurations using the mean F1 across validation folds.
    print("\n=== Ranking by mean CV F1 ===")
    for rank, result in enumerate(
        sorted(results, key=lambda item: item["mean_f1"], reverse=True),
        start=1,
    ):
        print(
            f"{rank}. {result['name']} {result['parameters']} - "
            f"F1={result['mean_f1']:.4f}, "
            f"threshold={result['threshold']:.6f}"
        )


if __name__ == "__main__":
    main()
