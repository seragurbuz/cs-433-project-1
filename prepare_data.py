
"""Clean raw BRFSS data and save CSV files for the trainer.

Run:
    python prepare_data.py --data dataset
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dataloader import load_data
from feature_rules import selected_features
from preprocessing import clean_data


def main(data_folder):
    data_folder = Path(data_folder)
    root = Path(__file__).resolve().parent

    # 1. Load the codebook and select the 250 features.
    codebook = json.loads(
        (root / "codebook_spec.json").read_text(encoding="utf-8")
    )
    feature_names = selected_features(codebook)

    # 2. Load the selected raw features and original IDs.
    x_train, x_test, _, _ = load_data(data_folder, feature_names)

    # 3. Clean BRFSS codes in both datasets.
    # Missing values remain NaN for the trainer to handle.
    x_train_clean = clean_data(x_train, feature_names, codebook)
    x_test_clean = clean_data(x_test, feature_names, codebook)

    # 4. Create the output folder.
    output_folder = data_folder / "cleaned_data"
    output_folder.mkdir(parents=True, exist_ok=True)

    # 5. Save feature-only files, IDs remain in the original raw CSVs
    header = ",".join(feature_names)

    np.savetxt(
        output_folder / "x_train_clean.csv",
        x_train_clean,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.10g",
    )

    np.savetxt(
        output_folder / "x_test_clean.csv",
        x_test_clean,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.10g",
    )

    print(f"Saved {len(feature_names)} features to {output_folder}")
    print(f"Training rows: {len(x_train_clean)}")
    print(f"Test rows: {len(x_test_clean)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset")
    args = parser.parse_args()

    main(args.data)
