"""Load the selected BRFSS features and training labels from CSV files."""

from pathlib import Path
import numpy as np

from helpers import load_csv_columns, load_csv_ids, load_labels_with_ids


def load_data(folder, feature_names):
    """Load selected train/test features, labels, and test IDs."""

    folder = Path(folder)

    # Load only the features selected in feature_rules.py
    # The same columns are loaded in the same order for train and test
    train, names = load_csv_columns(folder / "x_train.csv", feature_names)
    test, test_names = load_csv_columns(folder / "x_test.csv", feature_names)

    # Load IDs to check that features and labels are correctly aligned
    train_ids = load_csv_ids(folder / "x_train.csv")
    test_ids = load_csv_ids(folder / "x_test.csv")
    label_ids, labels = load_labels_with_ids(folder / "y_train.csv")

    # Check that train/test features match and labels belong to the right rows.
    if names != test_names or not np.array_equal(train_ids, label_ids):
        raise ValueError(
            "Train/test column names or training label IDs do not match"
        )

    # This is a binary classification problem with labels -1 and 1
    if not np.isin(labels, (-1, 1)).all():
        raise ValueError("Expected labels -1 and 1")

    return train, test, labels, test_ids
