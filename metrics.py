"""
Interchangeable classification metrics for labels in {-1, 1}.

Every public metric takes (y_true, y_pred) and returns a float; higher is
better. Inputs must be nonempty, matching one-dimensional label arrays.
The positive class is 1. Pass predicted labels, not probabilities or scores.

For example, on a held-out validation set:
    metric_function = f1_score
    y_pred = np.where(tx_validation @ w >= 0, 1, -1)
    score = metric_function(y_validation, y_pred)
"""

import numpy as np


def _confusion_counts(y_true, y_pred):
    """Validate labels and return true/false positive/negative counts."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    for labels in (y_true, y_pred):
        if labels.dtype.kind not in "iuf" or not np.isin(labels, (-1, 1)).all():
            raise ValueError("y_true and y_pred must contain only -1 and 1 labels")

    actual_positive = y_true == 1
    predicted_positive = y_pred == 1
    tp = int(np.count_nonzero(actual_positive & predicted_positive))
    fp = int(np.count_nonzero(~actual_positive & predicted_positive))
    fn = int(np.count_nonzero(actual_positive & ~predicted_positive))
    tn = int(np.count_nonzero(~actual_positive & ~predicted_positive))
    return tp, fp, fn, tn


def _safe_divide(numerator, denominator):
    """Return zero for an undefined ratio, otherwise a Python float."""
    return numerator / denominator if denominator else 0.0


def accuracy_score(y_true, y_pred):
    """Return the fraction of correctly predicted labels."""
    tp, fp, fn, tn = _confusion_counts(y_true, y_pred)
    return (tp + tn) / (tp + fp + fn + tn)


def precision_score(y_true, y_pred):
    """
    Return the fraction of predicted positives that are correct.

    The positive class is 1. Return 0.0 when no positives are predicted.
    """
    tp, fp, _, _ = _confusion_counts(y_true, y_pred)
    return _safe_divide(tp, tp + fp)


def recall_score(y_true, y_pred):
    """
    Return the fraction of actual positives that are detected.

    The positive class is 1. Return 0.0 when y_true has no positives.
    """
    tp, _, fn, _ = _confusion_counts(y_true, y_pred)
    return _safe_divide(tp, tp + fn)


def f1_score(y_true, y_pred):
    """
    Return the harmonic mean of positive-class precision and recall.

    The positive class is 1. Return 0.0 when there are no true positives,
    including when neither array contains any positives.
    """
    tp, fp, fn, _ = _confusion_counts(y_true, y_pred)
    return _safe_divide(2 * tp, 2 * tp + fp + fn)


def balanced_accuracy_score(y_true, y_pred):
    """
    Return the mean recall of the classes present in y_true.

    With both classes present, each receives equal weight regardless of
    class frequency. With only one true class, return that class's recall.
    """
    tp, fp, fn, tn = _confusion_counts(y_true, y_pred)
    recalls = []
    if tp + fn:
        recalls.append(tp / (tp + fn))
    if tn + fp:
        recalls.append(tn / (tn + fp))
    return sum(recalls) / len(recalls)
