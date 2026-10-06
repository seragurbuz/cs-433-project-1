"""Cross-validation experiments for the project models."""

from pathlib import Path

import numpy as np

from helpers import load_cleaned_data
from implementations import logistic_regression
from metrics import f1_score, precision_score, recall_score


DATA_PATH = Path(__file__).parent / "dataset"
CLEANED_TRAIN_PATH = DATA_PATH / "cleaned_data" / "x_train_replaced.csv"
CLEANED_TEST_PATH = DATA_PATH / "cleaned_data" / "x_test_replaced.csv"
LABELS_PATH = DATA_PATH / "y_train.csv"
TEST_IDS_PATH = DATA_PATH / "x_test.csv"

N_FOLDS = 5
SPLIT_SEED = 42
NUM_EPOCHS = 1000
GAMMA = 0.1


def _stratified_folds(y, n_folds, seed):
    """Return validation indices for reproducible stratified K-fold CV."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if n_folds > np.min(np.bincount((y == 1).astype(int))):
        raise ValueError("n_folds cannot exceed the size of the smallest class")

    rng = np.random.default_rng(seed)
    fold_indices = [[] for _ in range(n_folds)]
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        indices = rng.permutation(indices)
        for fold_number, fold_part in enumerate(np.array_split(indices, n_folds)):
            fold_indices[fold_number].extend(fold_part.tolist())

    return [rng.permutation(indices) for indices in fold_indices]


def _standardize(train_x, validation_x):
    """Standardize validation features with training-fold statistics."""
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0] = 1
    return (train_x - mean) / std, (validation_x - mean) / std


def _design_matrix(x):
    """Add an intercept column to standardized features."""
    return np.column_stack((np.ones(x.shape[0]), x))


def _fit_logistic_regression(x_train, y_train, num_epochs, gamma):
    """Fit logistic regression using the project's implementation."""
    tx_train = _design_matrix(x_train)
    y_train01 = (y_train + 1) / 2
    weights = np.zeros(tx_train.shape[1])
    for _ in range(num_epochs):
        weights, _ = logistic_regression(
            y_train01, tx_train, weights, max_iters=1, gamma=gamma
        )
    return weights


def _threshold_candidates(scores):
    """Return score thresholds covering the full predicted-positive range."""
    percentiles = np.percentile(scores, np.arange(0, 101))
    return np.unique(np.concatenate(([-np.inf, 0.0, np.inf], percentiles)))


def _best_threshold(validation_results):
    """Find the threshold with the highest mean F1 across validation folds."""
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
    return max(results, key=lambda result: (result[0], -result[1]))


def run_threshold_experiment(
    x, y, n_folds=N_FOLDS, seed=SPLIT_SEED, num_epochs=NUM_EPOCHS, gamma=GAMMA
):
    """Run stratified CV and select a threshold from out-of-fold scores."""
    validation_folds = _stratified_folds(y, n_folds, seed)
    out_of_fold_scores = np.empty(len(y))
    validation_results = []
    zero_threshold_f1_scores = []

    all_indices = np.arange(len(y))
    for fold_number, validation_indices in enumerate(validation_folds, start=1):
        training_indices = np.setdiff1d(all_indices, validation_indices, assume_unique=False)
        x_train, x_validation = _standardize(
            x[training_indices], x[validation_indices]
        )
        weights = _fit_logistic_regression(
            x_train, y[training_indices], num_epochs, gamma
        )
        validation_scores = _design_matrix(x_validation) @ weights
        out_of_fold_scores[validation_indices] = validation_scores
        validation_results.append((validation_scores, y[validation_indices]))

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
    print()
    print(
        f"Mean CV zero-threshold F1: "
        f"{np.mean(zero_threshold_f1_scores):.4f}"
    )
    print(f"OOF selected threshold: {best_threshold:.6f}")
    print(f"Mean CV selected F1: {best_f1:.4f}")
    print(
        f"Selected threshold fold F1: "
        f"{', '.join(f'{score:.4f}' for score in selected_fold_f1_scores)}"
    )
    print(f"OOF selected F1 (reference): {f1_score(y, best_predictions):.4f}")
    print(f"OOF precision: {precision_score(y, best_predictions):.4f}")
    print(f"OOF recall: {recall_score(y, best_predictions):.4f}")
    print(f"Recommended trainer setting: decision_threshold = {best_threshold:.6f}")
    return best_threshold, best_f1, selected_fold_f1_scores


def main():
    """Load the cleaned data and run the 5-fold threshold experiment."""
    x_train, _, y_train, _, _ = load_cleaned_data(
        CLEANED_TRAIN_PATH,
        CLEANED_TEST_PATH,
        LABELS_PATH,
        TEST_IDS_PATH,
    )
    print(
        f"Running {N_FOLDS}-fold logistic regression CV with "
        f"{NUM_EPOCHS} epochs and gamma={GAMMA}..."
    )
    run_threshold_experiment(x_train, y_train)


if __name__ == "__main__":
    main()
