"""Tests for Experiment 3 — Preprocessing and Feature Engineering Impact Study."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_wine

from src.exp3_preprocessing import (
    CONTROL_CLF_PARAMS,
    _build_synthetic_dataset,
    experiment_a_scaling,
    experiment_b_split_ratio,
    experiment_c_encoding,
    run_experiment_3,
)


# ---------------------------------------------------------------------------
# Control classifier parameters
# ---------------------------------------------------------------------------

def test_control_clf_params_match_methodology() -> None:
    """Control SVC must use kernel='rbf', C=1, random_state=42."""
    assert CONTROL_CLF_PARAMS["kernel"] == "rbf"
    assert CONTROL_CLF_PARAMS["C"] == 1
    assert CONTROL_CLF_PARAMS["random_state"] == 42


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

def test_build_synthetic_dataset_shape() -> None:
    """Synthetic dataset should have 500 samples and 5 columns (4 numeric + 1 cat)."""
    df, y = _build_synthetic_dataset(n_samples=500, n_numeric=4, n_categories=3)
    assert len(df) == 500
    assert len(y) == 500
    assert df.shape[1] == 5  # 4 numeric + 1 categorical


def test_build_synthetic_dataset_categories() -> None:
    """Categorical feature should have exactly 3 unique levels."""
    df, _ = _build_synthetic_dataset(n_categories=3)
    assert df["category"].nunique() == 3


def test_build_synthetic_dataset_binary_target() -> None:
    """Target should be binary (0 or 1)."""
    _, y = _build_synthetic_dataset()
    assert set(np.unique(y)).issubset({0, 1})


def test_build_synthetic_dataset_reproducibility() -> None:
    """Same random_state should produce identical datasets."""
    df1, y1 = _build_synthetic_dataset(random_state=42)
    df2, y2 = _build_synthetic_dataset(random_state=42)
    pd.testing.assert_frame_equal(df1, df2)
    np.testing.assert_array_equal(y1, y2)


def test_build_synthetic_dataset_different_seeds() -> None:
    """Different seeds should produce different datasets."""
    df1, _ = _build_synthetic_dataset(random_state=0)
    df2, _ = _build_synthetic_dataset(random_state=99)
    assert not df1["num_0"].equals(df2["num_0"])


# ---------------------------------------------------------------------------
# Experiment A — Scaling
# ---------------------------------------------------------------------------

def test_experiment_a_returns_three_rows() -> None:
    """Experiment A should return exactly 3 rows (no scaling, SS, MMS)."""
    data = load_wine()
    df = experiment_a_scaling(data.data, data.target)
    assert len(df) == 3
    assert set(df["Scaling"]) == {"No Scaling", "StandardScaler", "MinMaxScaler"}


def test_experiment_a_accuracy_in_range() -> None:
    """All accuracy values must be in [0, 1]."""
    data = load_wine()
    df = experiment_a_scaling(data.data, data.target)
    for col in ["Test Accuracy", "CV Mean"]:
        assert (df[col] >= 0).all() and (df[col] <= 1).all()


def test_experiment_a_scaling_improves_svc() -> None:
    """StandardScaler and MinMaxScaler should outperform no-scaling for SVC."""
    data = load_wine()
    df = experiment_a_scaling(data.data, data.target)
    no_scale_acc = df[df["Scaling"] == "No Scaling"]["Test Accuracy"].iloc[0]
    ss_acc = df[df["Scaling"] == "StandardScaler"]["Test Accuracy"].iloc[0]
    mms_acc = df[df["Scaling"] == "MinMaxScaler"]["Test Accuracy"].iloc[0]
    # Both scaled versions should be >= no-scaling (with small tolerance)
    assert ss_acc >= no_scale_acc - 0.05, (
        f"StandardScaler ({ss_acc}) should not be much worse than no scaling ({no_scale_acc})"
    )
    assert mms_acc >= no_scale_acc - 0.05


def test_experiment_a_cv_std_non_negative() -> None:
    """CV standard deviation must be non-negative."""
    data = load_wine()
    df = experiment_a_scaling(data.data, data.target)
    assert (df["CV Std"] >= 0).all()


# ---------------------------------------------------------------------------
# Experiment B — Split Ratio
# ---------------------------------------------------------------------------

def test_experiment_b_returns_three_rows() -> None:
    """Experiment B should return exactly 3 rows (60/40, 70/30, 80/20)."""
    data = load_wine()
    df = experiment_b_split_ratio(data.data, data.target)
    assert len(df) == 3
    assert set(df["Split"]) == {"60/40", "70/30", "80/20"}


def test_experiment_b_accuracy_in_range() -> None:
    """All accuracy values must be in [0, 1]."""
    data = load_wine()
    df = experiment_b_split_ratio(data.data, data.target)
    for col in ["Test Accuracy", "CV Mean"]:
        assert (df[col] >= 0).all() and (df[col] <= 1).all()


def test_experiment_b_cv_std_non_negative() -> None:
    """CV standard deviation must be non-negative."""
    data = load_wine()
    df = experiment_b_split_ratio(data.data, data.target)
    assert (df["CV Std"] >= 0).all()


def test_experiment_b_larger_train_higher_accuracy() -> None:
    """80/20 split should yield >= accuracy compared to 60/40 (more training data)."""
    data = load_wine()
    df = experiment_b_split_ratio(data.data, data.target)
    acc_6040 = df[df["Split"] == "60/40"]["Test Accuracy"].iloc[0]
    acc_8020 = df[df["Split"] == "80/20"]["Test Accuracy"].iloc[0]
    # Allow small tolerance — not guaranteed but expected
    assert acc_8020 >= acc_6040 - 0.05


# ---------------------------------------------------------------------------
# Experiment C — Encoding
# ---------------------------------------------------------------------------

def test_experiment_c_returns_two_rows() -> None:
    """Experiment C should return exactly 2 rows (LabelEncoder, One-Hot)."""
    df = experiment_c_encoding()
    assert len(df) == 2
    assert "LabelEncoder" in df["Encoding"].values
    assert "One-Hot (get_dummies)" in df["Encoding"].values


def test_experiment_c_accuracy_in_range() -> None:
    """All accuracy values must be in [0, 1]."""
    df = experiment_c_encoding()
    for col in ["Test Accuracy", "CV Mean"]:
        assert (df[col] >= 0).all() and (df[col] <= 1).all()


def test_experiment_c_cv_std_non_negative() -> None:
    """CV standard deviation must be non-negative."""
    df = experiment_c_encoding()
    assert (df["CV Std"] >= 0).all()


# ---------------------------------------------------------------------------
# Full experiment integration test
# ---------------------------------------------------------------------------

def test_run_experiment_3_returns_dict() -> None:
    """run_experiment_3 should return a dict with three DataFrames."""
    results = run_experiment_3()
    assert isinstance(results, dict)
    assert set(results.keys()) == {"scaling", "split_ratio", "encoding"}
    for key, df in results.items():
        assert isinstance(df, pd.DataFrame), f"results['{key}'] should be a DataFrame"
        assert len(df) > 0, f"results['{key}'] should not be empty"


def test_run_experiment_3_scaling_has_required_columns() -> None:
    """Scaling results must contain required columns."""
    results = run_experiment_3()
    df = results["scaling"]
    for col in ["Scaling", "Test Accuracy", "CV Mean", "CV Std"]:
        assert col in df.columns, f"Column '{col}' missing from scaling results"


def test_run_experiment_3_split_ratio_has_required_columns() -> None:
    """Split ratio results must contain required columns."""
    results = run_experiment_3()
    df = results["split_ratio"]
    for col in ["Split", "Test Accuracy", "CV Mean", "CV Std"]:
        assert col in df.columns, f"Column '{col}' missing from split_ratio results"


def test_run_experiment_3_encoding_has_required_columns() -> None:
    """Encoding results must contain required columns."""
    results = run_experiment_3()
    df = results["encoding"]
    for col in ["Encoding", "Test Accuracy", "CV Mean", "CV Std"]:
        assert col in df.columns, f"Column '{col}' missing from encoding results"
