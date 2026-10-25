"""Choose a model's decision threshold with cross-validation.

Run ``python experiments.py`` to prepare data if needed and select a threshold.
"""

from pathlib import Path
import hashlib
import json

import numpy as np

from helpers import (
    add_bias,
    apply_standardization,
    fit_standardization,
    load_labels_with_ids,
)
from implementations import (
    least_squares,
    logistic_regression,
    mean_squared_error_sgd,
    reg_logistic_regression,
    ridge_regression,
)
from preprocessing import fit_imputation, apply_imputation
from prepare_data import file_digest


def fit_model(model, x, labels, num_epochs, gamma, lambda_, seed):
    """Fit a course method to -1/1 labels with reproducible SGD sampling."""
    y = (
        (labels + 1) / 2
        if model in (logistic_regression, reg_logistic_regression)
        else labels
    )
    if model is least_squares:
        return model(y, x)
    if model is ridge_regression:
        return model(y, x, lambda_)
    if model is reg_logistic_regression:
        return model(y, x, lambda_, np.zeros(x.shape[1]), num_epochs, gamma)
    if model is mean_squared_error_sgd:
        np.random.seed(seed)
    return model(y, x, np.zeros(x.shape[1]), num_epochs, gamma)


def tune_training_threshold(x_train, y_train, config, model, model_dir, manifest):
    """Cache a threshold tied to the exact training split and model settings.

    The caller must reserve the outer holdout before passing training rows here.
    Neither holdout features nor labels are accepted by this function.
    """
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for values in (x_train, y_train):
        digest.update(str(values.shape).encode())
        digest.update(str(values.dtype).encode())
        for block in np.array_split(values, 64):
            digest.update(np.ascontiguousarray(block).tobytes())
    provenance = {
        "settings": config,
        "training_split_sha256": digest.hexdigest(),
        "preparation": manifest,
        "code": {
            name: file_digest(root / name)
            for name in (
                "experiments.py",
                "implementations.py",
                "helpers.py",
                "preprocessing.py",
            )
        },
    }
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "threshold_selection.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            record["provenance"] == provenance
            and np.isfinite(record["threshold"])
            and np.isfinite(record["oof_f1"])
        ):
            print(
                "Reusing threshold for the same training split and settings.",
                flush=True,
            )
            return record
    except (OSError, ValueError, KeyError, TypeError):
        pass
    threshold, score = select_cv_threshold(
        x_train,
        y_train,
        config["num_epochs"],
        config["gamma"],
        n_folds=config["n_folds"],
        seed=config["split_seed"],
        model=model,
        lambda_=config["lambda_"],
        plot_path=model_dir / "cv_threshold_diagnostics.png",
    )
    record = {
        "threshold": threshold,
        "oof_f1": score,
        "selection_scope": "training split only; outer holdout excluded",
        "training_rows": len(y_train),
        "provenance": provenance,
    }
    path.write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return record


def best_f1_threshold(scores, labels):
    """Find the score threshold with the highest F1 for labels in {-1, 1}.

    Args:
        scores: Raw model scores, one per sample
        labels: True -1/1 labels in the same order as scores

    Returns:
        The score threshold with the highest F1 and its F1 value
    """
    # order scores in descending order
    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    # order labels in the same order and mark positive classes as True
    positive = labels[order] == 1
    # count how many positive class predictions we would make for each possible score chosen as the threshold
    # and how many of those predictions are actually positive
    true_positives = np.cumsum(positive)
    predicted_positives = np.arange(1, len(scores) + 1)
    # mark the last item of each group having the same score as true
    boundaries = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    # calculate f1 after each sorted sample, only group boundaries are valid thresholds
    f1 = 2 * true_positives / (predicted_positives + positive.sum())
    # find the boundary with the highest f1 score and return its score and f1
    best = np.flatnonzero(boundaries)[np.argmax(f1[boundaries])]
    return float(sorted_scores[best]), float(f1[best])


def select_cv_threshold(
    x,
    labels,
    num_epochs,
    gamma,
    n_folds=5,
    seed=42,
    model=logistic_regression,
    lambda_=0.01,
    plot_path=None,
):
    """Fit on K-1 folds, then choose an F1 threshold from out-of-fold scores.

    Input x contains cleaned features with NaNs. Every fold learns its own
    imputation medians and standardization only from its training rows.

    Args:
        x: Cleaned feature matrix, missing values may still be NaN
        labels: True -1/1 labels aligned with the rows of x
        num_epochs: Number of updates for iterative models
        gamma: Learning rate for iterative models
        n_folds: Number of cross-validation folds
        seed: Random seed for fold assignment and SGD sampling
        model: Regression function to fit on each fold
        lambda_: Regularization strength for regularized models
        plot_path: Where to save the diagnostic plot, defaults to models/

    Returns:
        The selected raw-score threshold and its out-of-fold F1
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if not np.isin(labels, (-1, 1)).all():
        raise ValueError("labels must contain only -1 and 1")

    # Assign rows to folds, stratified by class
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_folds)]
    for label in (-1, 1):
        indices = rng.permutation(np.flatnonzero(labels == label))
        if len(indices) < n_folds:
            raise ValueError("Each class needs at least n_folds rows")
        for fold, part in zip(folds, np.array_split(indices, n_folds)):
            fold.extend(part)

    # allocate one validation score for each sample
    out_of_fold_scores = np.empty(len(labels))
    # In each iteration, hold out one fold and train on the other K-1 folds
    for fold_number, fold in enumerate(folds, start=1):
        validation_indices = np.asarray(fold, dtype=int)
        # True selects training rows, False indicates the validation rows for this fold
        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[validation_indices] = False

        x_train = x[train_mask]
        x_validation = x[validation_indices]
        # calculate medians from the training data
        # apply imputation to each fold's training and validation rows
        medians = fit_imputation(x_train)
        x_train = apply_imputation(x_train, medians)
        x_validation = apply_imputation(x_validation, medians)
        # fit scaling on training rows, then apply it to both sets
        mean, std = fit_standardization(x_train)
        tx_train = add_bias(apply_standardization(x_train, mean, std))
        tx_validation = add_bias(apply_standardization(x_validation, mean, std))

        weights, _ = fit_model(
            model,
            tx_train,
            labels[train_mask],
            num_epochs,
            gamma,
            lambda_,
            seed + fold_number,
        )
        # store each held-out score at its original row index
        out_of_fold_scores[validation_indices] = tx_validation @ weights
        print(f"CV fold {fold_number}/{n_folds} complete", flush=True)
        del x_train, x_validation, tx_train, tx_validation, weights

    # select the threshold with the best f1 from the out-of-fold scores
    threshold, best_f1 = best_f1_threshold(out_of_fold_scores, labels)
    # save a diagnostic plot of the threshold selection
    if plot_path is None:
        plot_path = (
            Path(__file__).resolve().parent / "models" / "cv_threshold_diagnostics.png"
        )
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from evaluation import save_threshold_diagnostics
    except ModuleNotFoundError as error:
        if error.name != "matplotlib":
            raise
        print("Matplotlib is unavailable; skipping the optional diagnostic plot.")
        return threshold, best_f1
    average_precision, selected_precision, selected_recall = save_threshold_diagnostics(
        out_of_fold_scores, labels, threshold, plot_path
    )
    print(
        f"OOF average precision: {average_precision:.4f}; "
        f"selected precision: {selected_precision:.4f}; "
        f"recall: {selected_recall:.4f}"
    )
    print(f"Saved threshold diagnostics to {plot_path}")
    return threshold, best_f1


def main():
    """Select a threshold for the model and training settings in trainer.py."""
    import trainer

    from helpers import split_training_validation

    x, _, labels, _, _, manifest = trainer.load_prepared_data()
    x_train, _, y_train, _ = split_training_validation(
        x, labels, trainer.validation_fraction, trainer.split_seed
    )
    record = tune_training_threshold(
        x_train,
        y_train,
        trainer.settings(),
        trainer.regression_function,
        trainer.model_dir,
        manifest,
    )
    print(
        f"Selected threshold: {record['threshold']:.6f}; tuning OOF F1: {record['oof_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
