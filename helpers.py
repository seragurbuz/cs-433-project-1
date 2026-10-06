"""Some helper functions for project 1."""

import csv
import numpy as np
import os
from pathlib import Path


def load_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsempled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # sub-sample
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids


def load_cleaned_data(cleaned_data_path, cleaned_test_path, labels_path, test_ids_path,
                      allow_missing=False):
    """
    Load feature-only train/test CSVs without running preprocessing.

    Labels and test IDs must have the same row order as the corresponding
    cleaned exports. Return (x_train, x_test, y_train, test_ids, feature_names).
    Paths may be strings or Path objects. Set allow_missing for feature-only
    exports before imputation; infinities are always rejected.
    """
    cleaned_data_path, cleaned_test_path, labels_path, test_ids_path = map(
        Path, (cleaned_data_path, cleaned_test_path, labels_path, test_ids_path)
    )
    # Cleaned exports contain features only; labels and IDs retain their original row order.
    for path in (cleaned_data_path, cleaned_test_path):
        if not path.is_file():
            raise FileNotFoundError(f"Cleaned data not found: {path}. Run dataloader.py first.")
    x_train = np.genfromtxt(cleaned_data_path, delimiter=",", skip_header=1, ndmin=2)
    x_test = np.genfromtxt(cleaned_test_path, delimiter=",", skip_header=1, ndmin=2)
    y_train_raw = np.genfromtxt(
        labels_path, delimiter=",", skip_header=1, usecols=1, ndmin=1
    )
    test_ids = np.genfromtxt(
        test_ids_path, delimiter=",", skip_header=1, usecols=0, dtype=int, ndmin=1
    )
    with cleaned_data_path.open(newline="") as source:
        feature_names = np.asarray(next(csv.reader(source)))
    if not len(x_train) or x_train.shape[1] != len(feature_names):
        raise ValueError("Cleaned feature columns must match the CSV header")
    if y_train_raw.shape != (len(x_train),) or not np.isin(y_train_raw, (-1, 1)).all():
        raise ValueError("Labels must contain -1/1 and match cleaned training rows")
    if np.isinf(x_train).any() or (not allow_missing and np.isnan(x_train).any()):
        raise ValueError("Cleaned data still contains NaN/inf; finish cleaning it first")
    with cleaned_test_path.open(newline="") as source:
        test_feature_names = next(csv.reader(source))
    if test_feature_names != feature_names.tolist():
        raise ValueError("Cleaned train/test CSVs must have identical feature headers")
    if x_test.shape != (len(test_ids), len(feature_names)):
        raise ValueError("Cleaned test rows/columns must match test IDs and training features")
    if np.isinf(x_test).any() or (not allow_missing and np.isnan(x_test).any()):
        raise ValueError("Cleaned test data still contains NaN/inf; finish cleaning it first")
    return x_train, x_test, y_train_raw, test_ids, feature_names


def split_training_validation(x, y, fraction, seed):
    """Split reproducibly, retaining each class in both sets (rounded per class)."""
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    training_indices, validation_indices = [], []
    for label in np.unique(y):
        indices = rng.permutation(np.flatnonzero(y == label))
        if len(indices) < 2:
            raise ValueError("Each class needs at least two rows for a stratified split")
        count = min(len(indices) - 1, max(1, round(len(indices) * fraction)))
        validation_indices.extend(indices[:count])
        training_indices.extend(indices[count:])
    training_indices = rng.permutation(training_indices)
    validation_indices = rng.permutation(validation_indices)
    return x[training_indices], x[validation_indices], y[training_indices], y[validation_indices]


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})
