"""Shared utilities for plotting, table formatting, and result persistence."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path(__file__).parent.parent / "results"


def ensure_results_dir() -> Path:
    """Create results directory if it does not exist and return its path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def save_figure(fig: plt.Figure, filename: str) -> Path:
    """Save a matplotlib figure to the results directory.

    Parameters
    ----------
    fig:
        Matplotlib figure to save.
    filename:
        Output filename (e.g. 'confusion_matrix.png').

    Returns
    -------
    Path
        Absolute path of the saved file.
    """
    out_dir = ensure_results_dir()
    path = out_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrices(
    cms: Dict[str, np.ndarray],
    class_names: List[str],
    title_prefix: str = "",
    filename: str = "confusion_matrices.png",
) -> Path:
    """Plot a grid of confusion matrix heatmaps.

    Parameters
    ----------
    cms:
        Mapping of model name → confusion matrix array.
    class_names:
        List of class label strings.
    title_prefix:
        Optional prefix for subplot titles.
    filename:
        Output filename.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_title(f"{title_prefix}{name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.tight_layout()
    return save_figure(fig, filename)


def plot_bar_comparison(
    df: pd.DataFrame,
    metric_cols: List[str],
    group_col: str,
    title: str,
    ylabel: str,
    filename: str,
) -> Path:
    """Plot grouped bar chart comparing metrics across conditions.

    Parameters
    ----------
    df:
        DataFrame with one row per condition.
    metric_cols:
        Column names to plot as grouped bars.
    group_col:
        Column used as x-axis labels.
    title:
        Chart title.
    ylabel:
        Y-axis label.
    filename:
        Output filename.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    x = np.arange(len(df))
    width = 0.8 / len(metric_cols)

    fig, ax = plt.subplots(figsize=(max(8, 2 * len(df)), 5))
    for i, col in enumerate(metric_cols):
        offset = (i - len(metric_cols) / 2 + 0.5) * width
        ax.bar(x + offset, df[col], width, label=col)

    ax.set_xticks(x)
    ax.set_xticklabels(df[group_col], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, filename)


def plot_scatter_actual_vs_predicted(
    y_true: np.ndarray,
    y_preds: Dict[str, np.ndarray],
    title: str = "Actual vs Predicted",
    filename: str = "scatter_actual_vs_predicted.png",
) -> Path:
    """Scatter plots of actual vs predicted values for multiple models.

    Parameters
    ----------
    y_true:
        Ground-truth target values.
    y_preds:
        Mapping of model name → predicted values.
    title:
        Overall figure title.
    filename:
        Output filename.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    n = len(y_preds)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, y_preds.items()):
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        lims = [
            min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max()),
        ]
        ax.plot(lims, lims, "r--", linewidth=1, label="Perfect fit")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(name)
        ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    return save_figure(fig, filename)


def append_results_md(section: str, content: str) -> None:
    """Append a section to results/RESULTS.md.

    Parameters
    ----------
    section:
        Section heading (e.g. 'Experiment 1 — Classification').
    content:
        Markdown-formatted content to append.
    """
    ensure_results_dir()
    path = RESULTS_DIR / "RESULTS.md"
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n## {section}\n\n{content}\n")


def init_results_md() -> None:
    """Initialise (overwrite) RESULTS.md with a header."""
    ensure_results_dir()
    path = RESULTS_DIR / "RESULTS.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Experiment Results\n\n")
        f.write("Generated by `run_experiments.py`.\n")
