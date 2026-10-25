"""Load the selected BRFSS features and training labels from CSV files."""

from pathlib import Path
import numpy as np

from helpers import load_csv_columns, load_csv_ids, load_labels_with_ids


def load_data(folder, feature_names):
    """Load selected features, training labels and test IDs from CSV files.

    Args:
        folder: Directory containing x_train.csv, x_test.csv, and y_train.csv
        feature_names: Selected feature columns to load

    Returns:
        x_train and x_test with the selected features, training labels aligned
        with x_train, and test IDs aligned with x_test
    """

    folder = Path(folder)

    # Load only the features selected in feature_rules.py
    # The same columns are loaded in the same order for train and test
    x_train = load_csv_columns(folder / "x_train.csv", feature_names)
    x_test = load_csv_columns(folder / "x_test.csv", feature_names)

    # Read IDs to verify that x_train rows match y_train labels in order
    train_ids = load_csv_ids(folder / "x_train.csv")
    # Keep test IDs to identify the rows when saving the predictions
    test_ids = load_csv_ids(folder / "x_test.csv")
    label_ids, labels = load_labels_with_ids(folder / "y_train.csv")

    # Check that labels belong to the right training rows
    if not np.array_equal(train_ids, label_ids):
        raise ValueError("Training feature and label IDs do not match")

    # This is a binary classification problem with labels -1 and 1
    # We do not expect to encounter a value different than -1 and 1, raise an error in such a case
    if not np.isin(labels, (-1, 1)).all():
        raise ValueError("Expected labels -1 and 1")

    return x_train, x_test, labels, test_ids
