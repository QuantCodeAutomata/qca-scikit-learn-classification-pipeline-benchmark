"""
Visualization Module
====================
Generates all plots for the classification benchmark experiment.
Saves figures to the results/ directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GridSearchCV

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_comparison_table(df: pd.DataFrame, save_path: Path | None = None) -> None:
    """Bar chart comparing classifiers by key metrics.

    Parameters
    ----------
    df : comparison DataFrame from build_comparison_table()
    save_path : optional path to save the figure
    """
    metrics = ["Test Accuracy", "Test Precision", "Test Recall", "Test F1"]
    x = np.arange(len(df))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, df[metric], width, label=metric)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df["Classifier"], rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Classifier Comparison — Test Set Metrics")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    path = save_path or RESULTS_DIR / "classifier_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_cv_scores(df: pd.DataFrame, save_path: Path | None = None) -> None:
    """Error-bar plot of cross-validation accuracy per classifier.

    Parameters
    ----------
    df : comparison DataFrame
    save_path : optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(
        df["Classifier"],
        df["CV Mean Accuracy"],
        yerr=df["CV Std"],
        fmt="o-",
        capsize=5,
        linewidth=2,
        markersize=8,
    )
    ax.set_ylim(0.8, 1.02)
    ax.set_ylabel("CV Accuracy (mean ± std)")
    ax.set_title("5-Fold Stratified Cross-Validation Accuracy")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()

    path = save_path or RESULTS_DIR / "cv_scores.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Path | None = None,
) -> None:
    """Heatmap confusion matrix for a classifier.

    Parameters
    ----------
    y_test : true labels
    y_pred : predicted labels
    title : plot title
    save_path : optional path to save the figure
    """
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()

    path = save_path or RESULTS_DIR / "confusion_matrix_best.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_gridsearch_heatmap(
    grid_search: GridSearchCV,
    save_path: Path | None = None,
) -> None:
    """Heatmap of GridSearchCV mean test scores for SVC.

    Parameters
    ----------
    grid_search : fitted GridSearchCV object
    save_path : optional path to save the figure
    """
    cv_results = pd.DataFrame(grid_search.cv_results_)
    pivot = cv_results.pivot_table(
        index="param_clf__kernel",
        columns="param_clf__C",
        values="mean_test_score",
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".4f",
        cmap="YlGnBu",
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("GridSearchCV — SVC Mean CV Accuracy\n(kernel × C)")
    ax.set_xlabel("C")
    ax.set_ylabel("Kernel")
    plt.tight_layout()

    path = save_path or RESULTS_DIR / "gridsearch_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_f1_ranking(df: pd.DataFrame, save_path: Path | None = None) -> None:
    """Horizontal bar chart ranking classifiers by F1-score.

    Parameters
    ----------
    df : comparison DataFrame sorted by F1 descending
    save_path : optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = sns.color_palette("viridis", len(df))
    bars = ax.barh(df["Classifier"][::-1], df["Test F1"][::-1], color=colors[::-1])
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Weighted F1-Score")
    ax.set_title("Classifier Ranking by Test F1-Score")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    path = save_path or RESULTS_DIR / "f1_ranking.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
