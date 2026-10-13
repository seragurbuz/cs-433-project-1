"""
Reproduce our final submission from the raw data.

Usage: python run.py
Expects dataset/x_train.csv, dataset/x_test.csv and dataset/y_train.csv
(unzip dataset.zip into dataset/). Writes submission.csv.
"""
from pathlib import Path

import numpy as np

from dataloader import load_clean_data
from helpers import add_bias, apply_standardization, create_csv_submission, fit_standardization
from implementations import (
    mean_squared_error_gd,
    mean_squared_error_sgd,
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
)

data_path = Path(__file__).parent / "dataset"
submission_path = Path(__file__).parent / "submission.csv"
# Selected by the regularization sweep at probability threshold 0.20.
# Select any of the six functions imported above.
regression_function = reg_logistic_regression
# Used by the iterative methods (GD, SGD, logistic); ignored by the closed-form ones.
num_epochs = 2000
gamma = 0.1
# Used only by ridge_regression and reg_logistic_regression.
lambda_ = 0.001
threshold = 0.2   # predict "sick" as soon as the probability is ≥ 20 %

def train(method, y, tx):
    """
    Train any of the six methods and return (w, loss).
    y must be in -1/1; logistic methods get it converted to 0/1.
    """
    if method in (logistic_regression, reg_logistic_regression):
        y = (y + 1) / 2
    initial_w = np.zeros(tx.shape[1])

    if method is least_squares:
        return least_squares(y, tx)
    if method is ridge_regression:
        return ridge_regression(y, tx, lambda_)
    if method is reg_logistic_regression:
        return reg_logistic_regression(y, tx, lambda_, initial_w, num_epochs, gamma)
    # mean_squared_error_gd, mean_squared_error_sgd, logistic_regression
    return method(y, tx, initial_w, num_epochs, gamma)


def main():
    # 1. Load the raw data, keep the selected features and replace missing values
    print("Loading and cleaning data...")
    x_train, x_test, y_train, test_ids, feature_names = load_clean_data(data_path)

    # 2. Standardize (mean/std learned on train only) and add the bias column
    mean, std = fit_standardization(x_train)
    tx_train = add_bias(apply_standardization(x_train, mean, std))
    tx_test = add_bias(apply_standardization(x_test, mean, std))

    # 3. Train the selected method
    print(f"Training {regression_function.__name__} on {tx_train.shape[0]} samples, "
          f"{len(feature_names)} features...")
    w, loss = train(regression_function, y_train, tx_train)
    print(f"Final training loss: {loss:.4f}")

    # 4. Predict on test and save. A score of 0 is the boundary for both families:
    # midpoint of -1/1 for linear methods, probability 0.5 for logistic ones.
    scores = tx_test @ w
    if regression_function in (logistic_regression, reg_logistic_regression):
        scores = 1 / (1 + np.exp(-scores))   # turn scores into probabilities
    y_pred = np.where(scores >= threshold, 1, -1)
    create_csv_submission(test_ids, y_pred, submission_path)
    print(f"Saved {submission_path} ({np.sum(y_pred == 1)} predicted positives out of {len(y_pred)})")


if __name__ == "__main__":
    main()
