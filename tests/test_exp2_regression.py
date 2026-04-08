"""
Tests for Experiment 2: Scikit-Learn Regression Pipeline Benchmark.

Validates:
- Data loading correctness
- Preprocessing properties
- Regressor training and prediction shapes
- Metric ranges and mathematical properties (RMSE = sqrt(MSE), R² ≤ 1)
- Comparison table structure
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exp2_regression import (
    build_comparison_table,
    build_regressors,
    compute_regression_metrics,
    load_california_housing_dataframe,
    preprocess_features,
    train_and_evaluate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def housing_data():
    X, y = load_california_housing_dataframe()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_train_scaled, X_test_scaled = preprocess_features(X_train, X_test)
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values


@pytest.fixture(scope="module")
def eval_results(housing_data):
    X_train_scaled, X_test_scaled, y_train, y_test = housing_data
    regressors = build_regressors()
    return train_and_evaluate(regressors, X_train_scaled, X_test_scaled, y_train, y_test)


# ---------------------------------------------------------------------------
# Data loading tests
# ---------------------------------------------------------------------------

def test_load_california_housing_shape():
    """California Housing must have 20640 samples and 8 features."""
    X, y = load_california_housing_dataframe()
    assert X.shape == (20640, 8), f"Expected (20640, 8), got {X.shape}"
    assert len(y) == 20640


def test_load_california_housing_no_missing():
    """California Housing must have no missing values."""
    X, y = load_california_housing_dataframe()
    assert X.isnull().sum().sum() == 0


def test_load_california_housing_target_continuous():
    """Target must be continuous (float dtype)."""
    _, y = load_california_housing_dataframe()
    assert y.dtype in [np.float32, np.float64, float]


def test_load_california_housing_target_positive():
    """All target values must be positive (house values)."""
    _, y = load_california_housing_dataframe()
    assert (y > 0).all()


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

def test_scaled_train_mean_near_zero():
    """StandardScaler must produce near-zero mean on training data."""
    X, y = load_california_housing_dataframe()
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.30, random_state=42)
    X_train_scaled, _ = preprocess_features(X_train, X_test)
    means = np.abs(X_train_scaled.mean(axis=0))
    assert np.all(means < 1e-10), f"Training means not near zero: {means}"


def test_scaled_train_std_near_one():
    """StandardScaler must produce near-unit std on training data."""
    X, y = load_california_housing_dataframe()
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.30, random_state=42)
    X_train_scaled, _ = preprocess_features(X_train, X_test)
    stds = X_train_scaled.std(axis=0)
    assert np.all(np.abs(stds - 1.0) < 1e-10), f"Training stds not near 1: {stds}"


def test_scaled_shapes_preserved():
    """Scaling must preserve the number of samples and features."""
    X, y = load_california_housing_dataframe()
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.30, random_state=42)
    X_train_scaled, X_test_scaled = preprocess_features(X_train, X_test)
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape


# ---------------------------------------------------------------------------
# Metric computation tests
# ---------------------------------------------------------------------------

def test_rmse_equals_sqrt_mse():
    """RMSE must equal sqrt(MSE) — mathematical property."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    metrics = compute_regression_metrics(y_true, y_pred)
    assert abs(metrics["rmse"] - np.sqrt(metrics["mse"])) < 1e-10, \
        "RMSE != sqrt(MSE)"


def test_r2_perfect_predictions():
    """R² must equal 1.0 for perfect predictions."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    metrics = compute_regression_metrics(y, y)
    assert abs(metrics["r2"] - 1.0) < 1e-10, f"R² for perfect preds = {metrics['r2']}"


def test_r2_at_most_one():
    """R² must be ≤ 1.0 for any predictions."""
    y_true = np.random.default_rng(42).normal(0, 1, 100)
    y_pred = np.random.default_rng(0).normal(0, 1, 100)
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["r2"] <= 1.0 + 1e-10


def test_mae_non_negative():
    """MAE must be non-negative."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["mae"] >= 0.0


def test_mse_non_negative():
    """MSE must be non-negative."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["mse"] >= 0.0


def test_mae_zero_for_perfect_predictions():
    """MAE must be 0 for perfect predictions."""
    y = np.array([1.0, 2.0, 3.0])
    metrics = compute_regression_metrics(y, y)
    assert abs(metrics["mae"]) < 1e-10


# ---------------------------------------------------------------------------
# Regressor tests
# ---------------------------------------------------------------------------

def test_build_regressors_returns_three():
    """build_regressors must return exactly 3 regressors."""
    regressors = build_regressors()
    assert len(regressors) == 3


def test_build_regressors_names():
    """Regressor names must match the methodology specification."""
    regressors = build_regressors()
    expected = {"LinearRegression", "Ridge", "RandomForestRegressor"}
    assert set(regressors.keys()) == expected


def test_predictions_shape(eval_results, housing_data):
    """Predictions must have the same length as the test set."""
    _, _, _, y_test = housing_data
    for name, res in eval_results.items():
        assert len(res["y_pred"]) == len(y_test), \
            f"{name}: prediction length mismatch"


def test_random_forest_best_r2(eval_results):
    """RandomForestRegressor must achieve the highest R² among the three models."""
    r2_scores = {name: res["metrics"]["r2"] for name, res in eval_results.items()}
    best_model = max(r2_scores, key=r2_scores.get)
    assert best_model == "RandomForestRegressor", \
        f"Expected RandomForestRegressor to have best R², got {best_model}: {r2_scores}"


def test_all_r2_positive(eval_results):
    """All regressors must achieve positive R² (better than mean baseline)."""
    for name, res in eval_results.items():
        r2 = res["metrics"]["r2"]
        assert r2 > 0, f"{name}: R²={r2:.4f} is not positive"


# ---------------------------------------------------------------------------
# Comparison table tests
# ---------------------------------------------------------------------------

def test_comparison_table_shape(eval_results):
    """Comparison table must have 3 rows (one per regressor)."""
    table = build_comparison_table(eval_results)
    assert table.shape[0] == 3


def test_comparison_table_columns(eval_results):
    """Comparison table must contain MAE, MSE, RMSE, R² columns."""
    table = build_comparison_table(eval_results)
    required_cols = {"MAE", "MSE", "RMSE", "R²"}
    assert required_cols.issubset(set(table.columns))


def test_comparison_table_rmse_equals_sqrt_mse(eval_results):
    """RMSE in comparison table must equal sqrt(MSE) for each regressor."""
    table = build_comparison_table(eval_results)
    for name in table.index:
        rmse = table.loc[name, "RMSE"]
        mse = table.loc[name, "MSE"]
        assert abs(rmse - np.sqrt(mse)) < 1e-3, \
            f"{name}: table RMSE {rmse} != sqrt(MSE) {np.sqrt(mse):.4f}"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

def test_metrics_with_constant_predictions():
    """Metrics must handle constant predictions without error."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.full(5, 3.0)  # constant prediction = mean
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["mae"] >= 0
    assert metrics["mse"] >= 0
    assert metrics["rmse"] >= 0


def test_metrics_with_single_sample():
    """Metrics must handle single-sample arrays without error."""
    y_true = np.array([2.5])
    y_pred = np.array([2.5])
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["mae"] == 0.0
    assert metrics["mse"] == 0.0
