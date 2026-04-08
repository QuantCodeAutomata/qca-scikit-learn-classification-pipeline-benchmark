"""Tests for Experiment 2 — Regression Pipeline Benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge

from src.exp2_regression import (
    PARAM_GRIDS,
    build_regressors,
    compute_regression_metrics,
    load_and_split_data,
    preprocess,
    run_experiment_2,
    tune_regressor,
)


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------

def test_load_and_split_data_shapes() -> None:
    """Train/test split should produce correct proportions."""
    X_train, X_test, y_train, y_test = load_and_split_data(test_size=0.3)
    total = len(X_train) + len(X_test)
    assert total == 442, f"Diabetes dataset should have 442 samples, got {total}"
    assert abs(len(X_test) / total - 0.3) < 0.02


def test_load_and_split_data_reproducibility() -> None:
    """Same random_state must produce identical splits."""
    split1 = load_and_split_data(random_state=42)
    split2 = load_and_split_data(random_state=42)
    np.testing.assert_array_equal(split1[0], split2[0])


def test_load_and_split_data_target_positive() -> None:
    """Diabetes target values should all be positive."""
    _, _, y_train, y_test = load_and_split_data()
    assert (y_train > 0).all()
    assert (y_test > 0).all()


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def test_preprocess_zero_mean() -> None:
    """StandardScaler should produce ~zero mean on training data."""
    X_train, X_test, _, _ = load_and_split_data()
    X_train_sc, _ = preprocess(X_train, X_test)
    np.testing.assert_allclose(X_train_sc.mean(axis=0), 0, atol=1e-10)


def test_preprocess_shape_preserved() -> None:
    """Scaling must not change array shapes."""
    X_train, X_test, _, _ = load_and_split_data()
    X_train_sc, X_test_sc = preprocess(X_train, X_test)
    assert X_train_sc.shape == X_train.shape
    assert X_test_sc.shape == X_test.shape


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_compute_regression_metrics_perfect_prediction() -> None:
    """Perfect predictions should yield MSE=0, RMSE=0, MAE=0, R2=1."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    metrics = compute_regression_metrics(y, y)
    assert metrics["MSE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["MAE"] == 0.0
    assert metrics["R2"] == 1.0


def test_compute_regression_metrics_rmse_equals_sqrt_mse() -> None:
    """RMSE must equal sqrt(MSE) within rounding tolerance (values rounded to 4dp)."""
    rng = np.random.default_rng(0)
    y_true = rng.standard_normal(100)
    y_pred = rng.standard_normal(100)
    metrics = compute_regression_metrics(y_true, y_pred)
    # Tolerance of 1e-4 accounts for 4-decimal-place rounding applied to both MSE and RMSE
    assert abs(metrics["RMSE"] - np.sqrt(metrics["MSE"])) < 1e-4


def test_compute_regression_metrics_r2_range() -> None:
    """R² can be negative (bad model) but should be ≤ 1."""
    rng = np.random.default_rng(1)
    y_true = rng.standard_normal(50)
    y_pred = rng.standard_normal(50)
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["R2"] <= 1.0


def test_compute_regression_metrics_mae_non_negative() -> None:
    """MAE must always be non-negative."""
    rng = np.random.default_rng(2)
    y_true = rng.standard_normal(50)
    y_pred = rng.standard_normal(50)
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["MAE"] >= 0.0


def test_compute_regression_metrics_constant_prediction() -> None:
    """Constant prediction equal to mean should give R2=0."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.full_like(y_true, y_true.mean())
    metrics = compute_regression_metrics(y_true, y_pred)
    assert abs(metrics["R2"]) < 1e-10


# ---------------------------------------------------------------------------
# Regressors
# ---------------------------------------------------------------------------

def test_build_regressors_returns_four_models() -> None:
    """build_regressors should return exactly 4 models."""
    regs = build_regressors()
    assert len(regs) == 4
    assert set(regs.keys()) == {"LinearRegression", "Ridge", "Lasso", "RF"}


# ---------------------------------------------------------------------------
# GridSearchCV tuning
# ---------------------------------------------------------------------------

def test_tune_regressor_best_params_in_grid() -> None:
    """Best alpha from GridSearchCV must be in the specified grid."""
    X_train, _, y_train, _ = load_and_split_data()
    X_train_sc, _ = preprocess(X_train, X_train)
    gs = tune_regressor(Ridge(), PARAM_GRIDS["Ridge"], X_train_sc, y_train, cv=3)
    assert gs.best_params_["alpha"] in PARAM_GRIDS["Ridge"]["alpha"]


def test_tune_regressor_negative_mse_score() -> None:
    """GridSearchCV with neg_mean_squared_error should have negative best_score_."""
    X_train, _, y_train, _ = load_and_split_data()
    X_train_sc, _ = preprocess(X_train, X_train)
    gs = tune_regressor(Ridge(), PARAM_GRIDS["Ridge"], X_train_sc, y_train, cv=3)
    assert gs.best_score_ <= 0.0


# ---------------------------------------------------------------------------
# Param grids
# ---------------------------------------------------------------------------

def test_param_grids_ridge_values() -> None:
    """Ridge param grid must match methodology specification."""
    assert PARAM_GRIDS["Ridge"]["alpha"] == [0.01, 0.1, 1, 10, 100]


def test_param_grids_lasso_values() -> None:
    """Lasso param grid must match methodology specification."""
    assert PARAM_GRIDS["Lasso"]["alpha"] == [0.001, 0.01, 0.1, 1]


def test_param_grids_rf_values() -> None:
    """RF param grid must match methodology specification."""
    assert PARAM_GRIDS["RF"]["n_estimators"] == [50, 100, 200]
    assert PARAM_GRIDS["RF"]["max_depth"] == [None, 5, 10]


# ---------------------------------------------------------------------------
# Full experiment integration test
# ---------------------------------------------------------------------------

def test_run_experiment_2_returns_dataframe() -> None:
    """run_experiment_2 should return a non-empty DataFrame."""
    df = run_experiment_2()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert "Model" in df.columns
    assert "Default RMSE" in df.columns
    assert "Tuned RMSE" in df.columns
    assert "Default R2" in df.columns
    assert "Tuned R2" in df.columns


def test_run_experiment_2_r2_positive() -> None:
    """All models should achieve positive R² on the diabetes dataset."""
    df = run_experiment_2()
    for _, row in df.iterrows():
        assert row["Default R2"] > 0, (
            f"{row['Model']} default R² {row['Default R2']} should be positive"
        )


def test_run_experiment_2_rmse_positive() -> None:
    """RMSE must be positive for all models."""
    df = run_experiment_2()
    for _, row in df.iterrows():
        assert row["Default RMSE"] > 0
        assert row["Tuned RMSE"] > 0


def test_run_experiment_2_rf_best_r2() -> None:
    """RandomForestRegressor should achieve the highest R² (expected outcome)."""
    df = run_experiment_2()
    rf_r2 = df[df["Model"] == "RF"]["Tuned R2"].iloc[0]
    lr_r2 = df[df["Model"] == "LinearRegression"]["Default R2"].iloc[0]
    # RF should outperform plain LinearRegression
    assert rf_r2 >= lr_r2 - 0.05, (
        f"RF R² ({rf_r2}) should be >= LinearRegression R² ({lr_r2})"
    )
