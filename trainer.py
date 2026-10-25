"""Tune on training folds, evaluate once on a holdout, then refit for submission."""

import json
from pathlib import Path

import numpy as np

from experiments import fit_model, tune_training_threshold
from helpers import (
    add_bias,
    apply_standardization,
    create_csv_submission,
    fit_standardization,
    load_cleaned_data,
    split_training_validation,
)
from implementations import (
    mean_squared_error_gd,
    mean_squared_error_sgd,
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
)
from metrics import accuracy_score, precision_score, recall_score, f1_score
from prepare_data import ensure_prepared_data
from preprocessing import fit_imputation, apply_imputation

ROOT = Path(__file__).resolve().parent
data_path = ROOT / "dataset"
model_dir = ROOT / "models"
submission_path = ROOT / "submission.csv"
regression_function = logistic_regression
num_epochs = 1000
gamma = 0.1
lambda_ = 0.01
validation_fraction = 0.2
split_seed = 42
n_folds = 5


def settings():
    """Serializable settings shared by standalone tuning and final training."""
    return dict(
        method=regression_function.__name__,
        num_epochs=num_epochs,
        gamma=gamma,
        lambda_=lambda_,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        n_folds=n_folds,
    )


def load_prepared_data():
    """Validate or regenerate cached features before any training entry point."""
    manifest = ensure_prepared_data(data_path)
    cleaned = Path(data_path) / "cleaned_data"
    arrays = load_cleaned_data(
        cleaned / "x_train_clean.csv",
        cleaned / "x_test_clean.csv",
        Path(data_path) / "y_train.csv",
        Path(data_path) / "x_test.csv",
        allow_missing=True,
    )
    return (*arrays, manifest)


def prepare_model_inputs(x_fit, x_other):
    """Learn imputation and scaling only on fitting rows and apply to both sets."""
    medians = fit_imputation(x_fit)
    fitted = apply_imputation(x_fit, medians)
    mean, std = fit_standardization(fitted)
    tx_fit = add_bias(apply_standardization(fitted, mean, std))
    tx_other = add_bias(
        apply_standardization(apply_imputation(x_other, medians), mean, std)
    )
    if not np.isfinite(tx_fit).all() or not np.isfinite(tx_other).all():
        raise ValueError("Preprocessing left nonfinite features")
    return tx_fit, tx_other, medians, mean, std


def main():
    """Execute the same reproducible workflow for run.py and direct invocation."""
    if num_epochs < 1 or gamma <= 0 or lambda_ < 0:
        raise ValueError("Require num_epochs >= 1, gamma > 0 and lambda_ >= 0")
    x_all, x_test, labels, test_ids, feature_names, manifest = load_prepared_data()
    x_train, x_holdout, y_train, y_holdout = split_training_validation(
        x_all, labels, validation_fraction, split_seed
    )
    print(
        f"Reserved {len(y_holdout)} holdout rows; tuning on {len(y_train)} training rows.",
        flush=True,
    )
    selection = tune_training_threshold(
        x_train, y_train, settings(), regression_function, model_dir, manifest
    )
    threshold = selection["threshold"]

    # The holdout is first scored after all model settings and the threshold are fixed.
    tx_train, tx_holdout, _, _, _ = prepare_model_inputs(x_train, x_holdout)
    w, _ = fit_model(
        regression_function, tx_train, y_train, num_epochs, gamma, lambda_, split_seed
    )
    predictions = np.where(tx_holdout @ w >= threshold, 1, -1)
    metrics = {
        name: float(fn(y_holdout, predictions))
        for name, fn in (
            ("f1", f1_score),
            ("accuracy", accuracy_score),
            ("precision", precision_score),
            ("recall", recall_score),
        )
    }
    print(f"Untouched holdout metrics: {metrics}", flush=True)
    del tx_train, tx_holdout, x_train, x_holdout

    # Keep the chosen threshold fixed while using every labeled row for the final fit.
    print(
        f"Refitting {regression_function.__name__} on all {len(labels)} rows...",
        flush=True,
    )
    tx_all, tx_test, medians, mean, std = prepare_model_inputs(x_all, x_test)
    w, loss = fit_model(
        regression_function, tx_all, labels, num_epochs, gamma, lambda_, split_seed
    )
    if not np.isfinite(w).all() or not np.isfinite(loss):
        raise ValueError("Training produced nonfinite weights or loss")
    model_path = Path(model_dir) / f"{regression_function.__name__}.npz"
    np.savez_compressed(
        model_path,
        weights=w,
        mean=mean,
        std=std,
        feature_names=feature_names,
        input_stage="codebook_cleaned_before_imputation",
        imputation_medians=medians,
        training_samples=len(labels),
        method=regression_function.__name__,
        threshold=threshold,
        threshold_selection=selection["selection_scope"],
        settings_json=json.dumps(settings()),
        fit_intercept=True,
        classes=np.array([-1, 1]),
    )
    report = dict(
        settings=settings(),
        threshold=threshold,
        tuning_oof_f1=selection["oof_f1"],
        holdout_metrics=metrics,
        holdout_rows=len(y_holdout),
        training_rows=len(labels),
        final_training_loss=float(loss),
        preparation=manifest,
    )
    (Path(model_dir) / "training_results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    create_csv_submission(
        test_ids, np.where(tx_test @ w >= threshold, 1, -1), submission_path
    )
    print(f"Saved {model_path} and {submission_path}", flush=True)
    return report


if __name__ == "__main__":
    main()
