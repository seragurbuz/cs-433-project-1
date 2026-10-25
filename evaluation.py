"""Plot cross-validation diagnostics for the positive (MICHD = 1) class."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_threshold_diagnostics(scores, labels, threshold, path):
    """Save plots of precision, recall and F1 across out-of-fold score thresholds.

    Args:
        scores: Raw model score for each sample
        labels: True -1/1 labels in the same order as scores
        threshold: Score at or above which a sample is predicted positive
        path: Where to save the plot

    Returns:
        Average precision, precision at the selected threshold and recall
        at the selected threshold
    """
    # order the scores in descending order (higher score means stronger prediction for positive class)
    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    # order the labels in the same order and mark positive class as True (negative class False)
    positive = labels[order] == 1
    total_positive = positive.sum()
    if total_positive == 0:
        raise ValueError("Precision-recall curve needs positive validation labels")
    # mark the last item of each group having the same score as true
    boundaries = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    # find how many positive class predictions we would make for each possible score chosen as the threshold
    # and how many of those predictions are actually positive
    # calculate precision and recall accordingly
    tp = np.cumsum(positive)[boundaries]
    predicted_positive = np.arange(1, len(scores) + 1)[boundaries]
    precision = tp / predicted_positive
    recall = tp / total_positive
    average_precision = np.sum(np.diff(np.r_[0, recall]) * precision)

    # mark the ones at or above the best selected threshold (selected in experiments.py) as positive class
    # calculate precision and recall with the best threshold
    selected = scores >= threshold
    selected_tp = np.count_nonzero(selected & (labels == 1))
    selected_precision = selected_tp / selected.sum() if selected.any() else 0.0
    selected_recall = selected_tp / total_positive

    # plot the PR curve and the threshold metrics side by side
    fig, (pr_ax, threshold_ax) = plt.subplots(1, 2, figsize=(14, 5))
    # left side shows the PR curve, positive-class prevalence and selected threshold
    pr_ax.step(np.r_[0, recall], np.r_[1, precision], where="pre", label="OOF PR")
    pr_ax.axhline(
        total_positive / len(labels),
        color="gray",
        linestyle="--",
        label="Positive class prevalence",
    )
    pr_ax.scatter(
        [selected_recall],
        [selected_precision],
        color="red",
        zorder=3,
        label=f"Selected threshold ({threshold:.3f})",
    )
    pr_ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Cross-validation PR curve",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    pr_ax.grid(alpha=0.25)
    pr_ax.legend()

    # reverse the thresholds so that they increase from left to right
    thresholds = sorted_scores[boundaries][::-1]
    f1 = 2 * tp / (predicted_positive + total_positive)
    # right side shows precision, recall and F1 at each threshold
    threshold_ax.plot(thresholds, precision[::-1], label="Precision")
    threshold_ax.plot(thresholds, recall[::-1], label="Recall")
    threshold_ax.plot(thresholds, f1[::-1], label="F1")
    threshold_ax.axvline(
        threshold,
        color="red",
        linestyle="--",
        label=f"Selected threshold ({threshold:.3f})",
    )
    threshold_ax.set(
        xlabel="Raw score threshold",
        ylabel="Metric",
        title="Metrics across thresholds",
        ylim=(0, 1),
    )
    threshold_ax.grid(alpha=0.25)
    threshold_ax.legend()
    # save the figure and close
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return float(average_precision), float(selected_precision), float(selected_recall)
