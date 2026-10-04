import csv

import numpy as np

from pathlib import Path

from helpers import create_csv_submission, load_csv_data
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
cleaned_data_path = data_path / "cleaned_data" / "x_train_replaced.csv"
cleaned_test_path = data_path / "cleaned_data" / "x_test_replaced.csv"
model_dir = Path(__file__).parent / "models"
# Select any of the six functions imported above.
regression_function = logistic_regression
# Select a metric imported above; all accept true and predicted -1/1 labels.
metric_function = f1_score
# Iterative methods take this many updates; SGD uses one sample per update.
# least_squares and ridge_regression solve once and ignore num_epochs/gamma.
num_epochs = 1000
gamma = 0.1
# Used only by ridge_regression and reg_logistic_regression.
lambda_ = 0.01  


def main():
    closed_form = regression_function in (least_squares, ridge_regression)
    is_logistic = regression_function in (logistic_regression, reg_logistic_regression)
    if not closed_form and num_epochs < 1:
        raise ValueError("num_epochs must be at least 1")
    if regression_function in (ridge_regression, reg_logistic_regression) and lambda_ < 0:
        raise ValueError("lambda_ must be non-negative")

    x_train, x_test, y_train_raw, _, test_ids = load_csv_data(data_path, cleaned=True)
    with cleaned_data_path.open(newline="") as source:
        feature_names = np.asarray(next(csv.reader(source)))
    if not len(x_train) or x_train.shape[1] != len(feature_names):
        raise ValueError("Cleaned feature columns must match the CSV header")
    if y_train_raw.shape != (len(x_train),) or not np.isin(y_train_raw, (-1, 1)).all():
        raise ValueError("Labels must contain -1/1 and match cleaned training rows")
    if not np.isfinite(x_train).all():
        raise ValueError("Cleaned data still contains NaN/inf; finish cleaning it first")
    with cleaned_test_path.open(newline="") as source:
        test_feature_names = next(csv.reader(source))
    if test_feature_names != feature_names.tolist():
        raise ValueError("Cleaned train/test CSVs must have identical feature headers")
    if x_test.shape != (len(test_ids), len(feature_names)):
        raise ValueError("Cleaned test rows/columns must match test IDs and training features")
    if not np.isfinite(x_test).all():
        raise ValueError("Cleaned test data still contains NaN/inf; finish cleaning it first")

    # The exported data is cleaned but unscaled. Fit scaling on training only.
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # Logistic methods need 0/1 labels; linear methods fit the original -1/1 labels.
    tx_train = np.column_stack((np.ones(x_train.shape[0]), x_train))
    tx_test = np.column_stack((np.ones(x_test.shape[0]), x_test))
    y_train = (y_train_raw + 1) / 2 if is_logistic else y_train_raw
    w = np.zeros(tx_train.shape[1])

    # Training process
    print(f"Training with {regression_function.__name__} on {tx_train.shape[0]} samples...")
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

        scores = tx_train @ w
        train_predictions = np.where(scores >= 0, 1, -1)
        metric_value = metric_function(y_train_raw, train_predictions)
        if closed_form:
            progress = "Closed-form solution"
        else:
            unit = "Iteration" if regression_function is mean_squared_error_sgd else "Epoch"
            progress = f"{unit} {step}/{num_epochs}"
        print(
            f"{progress} - "
            f"training loss: {loss:.4f}, "
            f"training {metric_function.__name__}: {metric_value:.4f}"
        )

    # This model consumes already-cleaned features in the saved header order.
    # A new run of the same method replaces its previous model file.
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{regression_function.__name__}.npz"
    np.savez_compressed(
        model_path,
        weights=w,
        mean=mean,
        std=std,
        feature_names=feature_names,
        input_stage="cleaned",
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
