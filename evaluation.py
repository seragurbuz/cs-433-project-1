"""Validation diagnostics for the positive (MICHD = 1) class."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_precision_recall_curve(scores, labels, threshold, path):
    """Plot validation precision/recall at every distinct score threshold."""
    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    positive = labels[order] == 1
    total_positive = positive.sum()
    if total_positive == 0:
        raise ValueError("Precision-recall curve needs positive validation labels")

    boundaries = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    tp = np.cumsum(positive)[boundaries]
    predicted_positive = np.arange(1, len(scores) + 1)[boundaries]
    precision = tp / predicted_positive
    recall = tp / total_positive
    average_precision = np.sum(np.diff(np.r_[0, recall]) * precision)

    selected = scores >= threshold
    selected_tp = np.count_nonzero(selected & (labels == 1))
    selected_precision = selected_tp / selected.sum() if selected.any() else 0.0
    selected_recall = selected_tp / total_positive

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(np.r_[0, recall], np.r_[1, precision], where="pre", label="Validation PR")
    ax.axhline(total_positive / len(labels), color="gray", linestyle="--",
               label="Positive class prevalence")
    ax.scatter([selected_recall], [selected_precision], color="red", zorder=3,
               label=f"Selected F1 threshold ({threshold:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Validation precision–recall curve",
           xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return float(average_precision), float(selected_precision), float(selected_recall)
