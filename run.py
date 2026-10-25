"""Run the complete BRFSS pipeline.

Usage:
    python run.py

Reads raw data from dataset/ and creates submission.csv
"""

from trainer import main as train_model


def main():
    # The trainer validates cached preprocessing, tunes, evaluates and refits.
    return train_model()


if __name__ == "__main__":
    main()
