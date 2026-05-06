"""
eval_metrics.py
Shared evaluation helpers used by both training scripts and the ablation.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, confusion_matrix,
    roc_curve, auc
)


def compute_f1_macro(labels, preds, n_classes):
    return f1_score(labels, preds, average="macro",
                    labels=list(range(n_classes)),
                    zero_division=0)


def confusion_matrix_plot(labels, preds, class_names,
                           save_path, title="Confusion matrix"):
    cm  = confusion_matrix(labels, preds,
                           labels=list(range(len(class_names))))
    cmn = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cmn, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title(title,        fontsize=13)

    thresh = 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cmn[i,j]:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if cmn[i,j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def roc_curves_plot(labels, probs, class_names, save_path,
                    title="ROC curves"):
    """One-vs-rest ROC curve per class."""
    n = len(class_names)
    labels_oh = np.eye(n)[labels]   # one-hot

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, n))

    for i, (name, col) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(labels_oh[:, i], probs[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=1.5,
                label=f"{name} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def training_curve_plot(log_csv, save_path, metric="val_f1", title=""):
    """Plot a metric column from a training log CSV."""
    import csv
    epochs, values = [], []
    with open(log_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            values.append(float(row[metric]))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, values, marker="o", markersize=3, lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or metric)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")
