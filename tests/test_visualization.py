"""
Tests for the visualization module.
Validates that plots are generated without errors and saved to disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pipeline import ClassifierResult, build_comparison_table, run_experiment
from src.visualization import (
    plot_comparison_table,
    plot_confusion_matrix,
    plot_cv_scores,
    plot_f1_ranking,
    plot_gridsearch_heatmap,
)


@pytest.fixture(scope="module")
def sample_comparison_df():
    results = [
        ClassifierResult("KNN", 0.92, 0.91, 0.92, 0.91, 0.90, 0.02),
        ClassifierResult("SVC", 0.97, 0.97, 0.97, 0.97, 0.96, 0.01),
        ClassifierResult("GaussianNB", 0.88, 0.87, 0.88, 0.87, 0.86, 0.03),
        ClassifierResult("RandomForest", 0.96, 0.96, 0.96, 0.96, 0.95, 0.01),
    ]
    return build_comparison_table(results)


@pytest.fixture(scope="module")
def experiment_output():
    return run_experiment()


def test_plot_comparison_table_saves_file(sample_comparison_df, tmp_path):
    """plot_comparison_table must save a PNG file."""
    out = tmp_path / "comparison.png"
    plot_comparison_table(sample_comparison_df, save_path=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_cv_scores_saves_file(sample_comparison_df, tmp_path):
    """plot_cv_scores must save a PNG file."""
    out = tmp_path / "cv.png"
    plot_cv_scores(sample_comparison_df, save_path=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_confusion_matrix_saves_file(tmp_path):
    """plot_confusion_matrix must save a PNG file."""
    y_test = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0])
    out = tmp_path / "cm.png"
    plot_confusion_matrix(y_test, y_pred, save_path=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_gridsearch_heatmap_saves_file(experiment_output, tmp_path):
    """plot_gridsearch_heatmap must save a PNG file."""
    gs = experiment_output["grid_search"]
    out = tmp_path / "heatmap.png"
    plot_gridsearch_heatmap(gs, save_path=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_f1_ranking_saves_file(sample_comparison_df, tmp_path):
    """plot_f1_ranking must save a PNG file."""
    out = tmp_path / "f1.png"
    plot_f1_ranking(sample_comparison_df, save_path=out)
    assert out.exists()
    assert out.stat().st_size > 0
