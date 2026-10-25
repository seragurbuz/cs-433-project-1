"""Clean raw BRFSS data and save CSV files for the trainer.

Run:
    python prepare_data.py --data dataset
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from dataloader import load_data
from feature_rules import selected_features
from preprocessing import clean_data


def file_digest(path):
    """Hash a file without loading the entire dataset into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preparation_signature(data_folder):
    """Identify raw inputs and all code defining the cleaned feature schema."""
    root = Path(__file__).resolve().parent
    return {
        "raw": {
            name: file_digest(Path(data_folder) / name)
            for name in ("x_train.csv", "x_test.csv", "y_train.csv")
        },
        "code": {
            name: file_digest(root / name)
            for name in (
                "codebook_spec.json",
                "feature_rules.py",
                "preprocessing.py",
                "dataloader.py",
                "helpers.py",
                "prepare_data.py",
            )
        },
    }


def ensure_prepared_data(data_folder):
    """Reuse only cleaned CSVs matching the current inputs, code and headers."""
    data_folder = Path(data_folder)
    output = data_folder / "cleaned_data"
    signature = preparation_signature(data_folder)
    root = Path(__file__).resolve().parent
    names = selected_features(
        json.loads((root / "codebook_spec.json").read_text(encoding="utf-8"))
    )
    try:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        valid = manifest["signature"] == signature
        for name in ("x_train_clean.csv", "x_test_clean.csv"):
            with (output / name).open(newline="", encoding="utf-8") as source:
                valid &= next(csv.reader(source)) == names
            valid &= manifest["outputs"][name] == file_digest(output / name)
        if valid:
            return manifest
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        pass
    print("Cleaned data is missing or outdated; rebuilding...", flush=True)
    main(data_folder)
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


def main(data_folder):
    data_folder = Path(data_folder)
    root = Path(__file__).resolve().parent
    signature = preparation_signature(data_folder)

    # 1. Load the codebook and select the 250 features.
    codebook = json.loads((root / "codebook_spec.json").read_text(encoding="utf-8"))
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
    manifest = {
        "signature": signature,
        "feature_names": feature_names,
        "outputs": {
            name: file_digest(output_folder / name)
            for name in ("x_train_clean.csv", "x_test_clean.csv")
        },
    }
    (output_folder / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset")
    args = parser.parse_args()

    main(args.data)
