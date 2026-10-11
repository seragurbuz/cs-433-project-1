"""Train, validate and submit using cleaned BRFSS CSV files.

Run prepare_data.py first. Keep run.py unchanged.
"""
import json
import numpy as np
from pathlib import Path

from helpers import create_csv_submission, load_cleaned_data, split_training_validation
from preprocessing import fit_imputation, apply_imputation
from implementations import (
    mean_squared_error_gd, mean_squared_error_sgd, least_squares,
    ridge_regression, logistic_regression, reg_logistic_regression,
)
from metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score,
)

data_path = Path(__file__).parent / "dataset"
cleaned_data_path = data_path / "cleaned_data" / "x_train_clean.csv"
cleaned_test_path = data_path / "cleaned_data" / "x_test_clean.csv"
model_dir = Path(__file__).parent / "models"
regression_function = logistic_regression
metric_function = f1_score
num_epochs = 1000
training_print_interval = 10
validation_fraction = 0.2
validation_interval = 100
split_seed = 42
gamma = 0.1
lambda_ = 0.01


def find_best_f1_threshold(scores, labels):
    """Return the score threshold that maximizes validation F1."""

    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    positive = labels[order] == 1
    true_positives = np.cumsum(positive)
    false_positives = np.cumsum(~positive)
    false_negatives = true_positives[-1] - true_positives
    boundaries = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    f1_values = np.zeros_like(sorted_scores, dtype=float)
    denominator = 2 * true_positives + false_positives + false_negatives
    f1_values[boundaries] = (
        2 * true_positives[boundaries] / denominator[boundaries]
    )
    best_index = np.argmax(f1_values)
    return float(sorted_scores[best_index]), float(f1_values[best_index])


def main():
    closed_form = regression_function in (least_squares, ridge_regression)
    is_logistic = regression_function in (logistic_regression, reg_logistic_regression)
    if not closed_form and num_epochs < 1:
        raise ValueError("num_epochs must be at least 1")
    if not isinstance(training_print_interval, (int, np.integer)) or training_print_interval < 1:
        raise ValueError("training_print_interval must be a positive integer")
    if not isinstance(validation_interval, (int, np.integer)) or validation_interval < 1:
        raise ValueError("validation_interval must be a positive integer")
    if regression_function in (ridge_regression, reg_logistic_regression) and lambda_ < 0:
        raise ValueError("lambda_ must be non-negative")

    x_all, x_test, y_all_raw, test_ids, feature_names = load_cleaned_data(
        cleaned_data_path, cleaned_test_path,
        data_path / "y_train.csv", data_path / "x_test.csv", allow_missing=True,
    )

    # BRFSS special codes were already converted to NaN in prepare_data.py.
    # Only fill NaNs here; learn medians on the training split to avoid leakage.
    # This deliberately does NOT reuse old raw-code REPLACEMENTS: a value such
    # as 7 can be a legitimate value in a different cleaned feature.
    # Split before filling missing values or scaling.
    x_train, x_validation, y_train_raw, y_validation_raw = split_training_validation(
        x_all, y_all_raw, validation_fraction, split_seed
    )
    medians = fit_imputation(x_train)
    x_train = apply_imputation(x_train, medians)
    x_validation = apply_imputation(x_validation, medians)
    if not np.isfinite(x_train).all() or not np.isfinite(x_validation).all():
        raise ValueError("Replacement rules left missing/nonfinite features")
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1
    x_train = (x_train - mean) / std
    x_validation = (x_validation - mean) / std

    # Logistic methods need 0/1 labels; linear methods fit original -1/1 labels.
    tx_train = np.column_stack((np.ones(x_train.shape[0]), x_train))
    tx_validation = np.column_stack((np.ones(x_validation.shape[0]), x_validation))
    y_train = (y_train_raw + 1) / 2 if is_logistic else y_train_raw
    y_validation = (y_validation_raw + 1) / 2 if is_logistic else y_validation_raw
    w = np.zeros(tx_train.shape[1])

    # Training process
    print(f"Training with {regression_function.__name__} on {tx_train.shape[0]} samples...")
    print(f"Holding out {tx_validation.shape[0]} samples for validation (split seed: {split_seed}).")
    training_steps = 1 if closed_form else num_epochs
    for step in range(1, training_steps + 1):
        if regression_function is least_squares:
            w, loss = regression_function(y_train, tx_train)
        elif regression_function is ridge_regression:
            w, loss = regression_function(y_train, tx_train, lambda_)
        elif regression_function is reg_logistic_regression:
            w, loss = regression_function(y_train, tx_train, lambda_, w, 1, gamma)
        else:
            w, loss = regression_function(y_train, tx_train, w, 1, gamma)

        if closed_form:
            progress = "Closed-form solution"
        else:
            unit = "Iteration" if regression_function is mean_squared_error_sgd else "Epoch"
            progress = f"{unit} {step}/{num_epochs}"
        if step % training_print_interval == 0 or step == training_steps:
            scores = tx_train @ w
            train_predictions = np.where(scores >= 0, 1, -1)
            metric_value = metric_function(y_train_raw, train_predictions)
            print(
                f"{progress} - "
                f"training loss: {loss:.4f}, "
                f"training {metric_function.__name__}: {metric_value:.4f}"
            )
        if step % validation_interval == 0 or step == training_steps:
            validation_scores = tx_validation @ w
            if is_logistic:
                validation_loss = np.mean(
                    np.logaddexp(0, validation_scores) - y_validation * validation_scores
                )
            else:
                validation_loss = 0.5 * np.mean((y_validation - validation_scores) ** 2)
            validation_predictions = np.where(validation_scores >= 0, 1, -1)
            validation_metric = metric_function(y_validation_raw, validation_predictions)
            print(
                f"{progress} - validation loss: {validation_loss:.4f}, "
                f"validation {metric_function.__name__}: {validation_metric:.4f}"
            )

    validation_scores = tx_validation @ w
    decision_threshold, validation_metric = find_best_f1_threshold(
        validation_scores, y_validation_raw
    )
    print(
        f"Selected decision threshold: {decision_threshold:.6f} "
        f"(validation {metric_function.__name__}: {validation_metric:.4f})"
    )

    # Retrain on all labeled rows for the final submission.
    print(f"Refitting {regression_function.__name__} on all {len(x_all)} labeled samples...")
    medians = fit_imputation(x_all)
    x_all = apply_imputation(x_all, medians)
    x_test = apply_imputation(x_test, medians)
    if not np.isfinite(x_all).all() or not np.isfinite(x_test).all():
        raise ValueError("Imputation left missing/nonfinite features")
    mean = x_all.mean(axis=0)
    std = x_all.std(axis=0)
    std[std == 0] = 1
    tx_all = np.column_stack((np.ones(len(x_all)), (x_all - mean) / std))
    tx_test = np.column_stack((np.ones(len(x_test)), (x_test - mean) / std))
    y_all = (y_all_raw + 1) / 2 if is_logistic else y_all_raw
    if regression_function is least_squares:
        w, loss = regression_function(y_all, tx_all)
    elif regression_function is ridge_regression:
        w, loss = regression_function(y_all, tx_all, lambda_)
    elif regression_function is reg_logistic_regression:
        w, loss = regression_function(
            y_all, tx_all, lambda_, np.zeros(tx_all.shape[1]), num_epochs, gamma
        )
    else:
        w, loss = regression_function(
            y_all, tx_all, np.zeros(tx_all.shape[1]), num_epochs, gamma
        )
    print(f"Full-data refit complete - training loss: {loss:.4f}")

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{regression_function.__name__}.npz"
    np.savez_compressed(
        model_path,
        weights=w,
        mean=mean,
        std=std,
        feature_names=feature_names,
        input_stage="selected_before_replacement",
        imputation_medians=medians,
        training_samples=len(y_all),
        method=regression_function.__name__,
        threshold=decision_threshold,
        fit_intercept=True,
        classes=np.array([-1, 1]),
    )
    print(f"Saved model to {model_path}")

    test_predictions = np.where(tx_test @ w >= decision_threshold, 1, -1)
    create_csv_submission(test_ids, test_predictions, "submission.csv")
    print("Saved submission.csv")


if __name__ == "__main__":
    main()
