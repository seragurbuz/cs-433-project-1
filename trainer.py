import json
import numpy as np

from pathlib import Path

from helpers import create_csv_submission, load_cleaned_data, split_training_validation
from dataloader import REPLACEMENTS, fit_replacements, replace_values
from implementations import (
    mean_squared_error_gd,
    mean_squared_error_sgd,
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
)
from metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)

data_path = Path(__file__).parent / "dataset"
cleaned_data_path = data_path / "cleaned_data" / "x_train_clean.csv"
cleaned_test_path = data_path / "cleaned_data" / "x_test_clean.csv"
model_dir = Path(__file__).parent / "models"
# Select any of the six functions imported above.
regression_function = logistic_regression
# Select a metric imported above; all accept true and predicted -1/1 labels.
metric_function = f1_score
# Iterative methods take this many updates; SGD uses one sample per update.
# least_squares and ridge_regression solve once and ignore num_epochs/gamma.
num_epochs = 1000
# Print training results every this many updates and at the final step.
training_print_interval = 10
# Fraction held out for validation; the remaining rows are used for training.
validation_fraction = 0.2
validation_interval = 100
split_seed = 42
gamma = 0.1
# Used only by ridge_regression and reg_logistic_regression.
lambda_ = 0.01  


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
    # These exports only select columns; split before learning any statistics.
    x_train, x_validation, y_train_raw, y_validation_raw = split_training_validation(
        x_all, y_all_raw, validation_fraction, split_seed
    )
    rules = fit_replacements(x_train, feature_names, REPLACEMENTS)
    x_train = replace_values(x_train, feature_names, rules)
    x_validation = replace_values(x_validation, feature_names, rules)
    if not np.isfinite(x_train).all() or not np.isfinite(x_validation).all():
        raise ValueError("Replacement rules left missing/nonfinite features")
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1
    x_train = (x_train - mean) / std
    x_validation = (x_validation - mean) / std

    # Logistic methods need 0/1 labels; linear methods fit the original -1/1 labels.
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
            # One update, starting from the previous weights (one sample for SGD).
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
        # Evaluate without updating weights, including the final/closed-form result.
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

    # Validation above estimates performance. Refit from scratch on all labeled
    # rows with the same configuration for the final model and submission.
    print(f"Refitting {regression_function.__name__} on all {len(x_all)} labeled samples...")
    rules = fit_replacements(x_all, feature_names, REPLACEMENTS)
    x_all = replace_values(x_all, feature_names, rules)
    x_test = replace_values(x_test, feature_names, rules)
    if not np.isfinite(x_all).all() or not np.isfinite(x_test).all():
        raise ValueError("Replacement rules left missing/nonfinite features")
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

    # Save numeric replacement rules alongside full-data scaling and weights.
    # JSON null represents a missing-value (NaN) matching rule.
    replacement_records = [
        [name, None if np.isnan(old) else float(old), float(new)]
        for name, feature_rules in rules.items() for old, new in feature_rules.items()
    ]
    # This model consumes selected features before replacement, in saved order.
    # A new run of the same method replaces its previous model file.
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{regression_function.__name__}.npz"
    np.savez_compressed(
        model_path,
        weights=w,
        mean=mean,
        std=std,
        feature_names=feature_names,
        input_stage="selected_before_replacement",
        replacement_rules_json=json.dumps(replacement_records, allow_nan=False),
        training_samples=len(y_all),
        method=regression_function.__name__,
        threshold=0.0,
        fit_intercept=True,
        classes=np.array([-1, 1]),
    )
    print(f"Saved model to {model_path}")

    # Zero is the boundary for -1/1 regression and a logistic probability of 0.5.
    test_predictions = np.where(tx_test @ w >= 0, 1, -1)
    create_csv_submission(test_ids, test_predictions, "submission.csv")
    print("Saved submission.csv")


if __name__ == "__main__":
    main()
