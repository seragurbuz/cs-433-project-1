"""Choose a model's decision threshold with cross-validation."""

import numpy as np

from implementations import (
    least_squares, logistic_regression, mean_squared_error_sgd,
    reg_logistic_regression, ridge_regression,
)
from preprocessing import fit_imputation, apply_imputation


def best_f1_threshold(scores, labels):
    """Find the score threshold with the highest F1 for labels in {-1, 1}."""
    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    positive = labels[order] == 1
    true_positives = np.cumsum(positive)
    predicted_positives = np.arange(1, len(scores) + 1)
    boundaries = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    f1 = 2 * true_positives / (predicted_positives + positive.sum())
    best = np.flatnonzero(boundaries)[np.argmax(f1[boundaries])]
    return float(sorted_scores[best]), float(f1[best])


def select_cv_threshold(
    x, labels, num_epochs, gamma, n_folds=5, seed=42,
    model=logistic_regression, lambda_=0.01,
):
    """Fit on K-1 folds, score the held-out fold, then tune F1 on all OOF scores.

    Input x contains cleaned features with NaNs. Every fold learns its own
    imputation medians and standardization from its training rows only.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if not np.isin(labels, (-1, 1)).all():
        raise ValueError("labels must contain only -1 and 1")

    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_folds)]
    for label in (-1, 1):
        indices = rng.permutation(np.flatnonzero(labels == label))
        if len(indices) < n_folds:
            raise ValueError("Each class needs at least n_folds rows")
        for fold, part in zip(folds, np.array_split(indices, n_folds)):
            fold.extend(part)

    out_of_fold_scores = np.empty(len(labels))
    for fold_number, fold in enumerate(folds, start=1):
        validation_indices = np.asarray(fold, dtype=int)
        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[validation_indices] = False

        x_train = x[train_mask]
        x_validation = x[validation_indices]
        medians = fit_imputation(x_train)
        x_train = apply_imputation(x_train, medians)
        x_validation = apply_imputation(x_validation, medians)
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        std[std == 0] = 1
        tx_train = np.column_stack((np.ones(len(x_train)), (x_train - mean) / std))
        tx_validation = np.column_stack(
            (np.ones(len(x_validation)), (x_validation - mean) / std)
        )

        y_train = labels[train_mask]
        if model in (logistic_regression, reg_logistic_regression):
            y_train = (y_train + 1) / 2
        if model is least_squares:
            weights, _ = model(y_train, tx_train)
        elif model is ridge_regression:
            weights, _ = model(y_train, tx_train, lambda_)
        elif model is reg_logistic_regression:
            weights, _ = model(
                y_train, tx_train, lambda_, np.zeros(tx_train.shape[1]),
                num_epochs, gamma,
            )
        else:
            if model is mean_squared_error_sgd:
                np.random.seed(seed + fold_number)
            weights, _ = model(
                y_train, tx_train, np.zeros(tx_train.shape[1]), num_epochs, gamma
            )
        out_of_fold_scores[validation_indices] = tx_validation @ weights
        print(f"CV fold {fold_number}/{n_folds} complete", flush=True)
        del x_train, x_validation, tx_train, tx_validation, weights

    return best_f1_threshold(out_of_fold_scores, labels)
