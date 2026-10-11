"""Run the complete BRFSS pipeline.

Usage:
    python run.py

Reads raw data from dataset/ and creates submission.csv.
"""

from pathlib import Path

from prepare_data import main as prepare_data
from trainer import main as train_model


def main():
    data_path = Path(__file__).parent / "dataset"

    # 1. Clean the raw BRFSS data and save cleaned CSV files.
    print("Preparing data...")
    prepare_data(data_path)

    # 2. Train, validate, refit on all labeled data,
    #    and generate submission.csv.
    print("Training model...")
    train_model()


if __name__ == "__main__":
    main()
