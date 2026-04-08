"""Tests for Experiment 1 — Classification Pipeline Benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from src.exp1_classification import (
    PARAM_GRIDS,
    build_classifiers,
    evaluate_classifier,
    load_and_split_data,
    preprocess,
    run_experiment_1,
    tune_classifier,
)


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------

def test_load_and_split_data_shapes() -> None:
    """Train/test split should produce correct proportions."""
    X_train, X_test, y_train, y_test = load_and_split_data(test_size=0.3)
    total = len(X_train) + len(X_test)
    assert total == 569, f"Breast cancer dataset should have 569 samples, got {total}"
    assert abs(len(X_test) / total - 0.3) < 0.02, "Test set should be ~30% of data"


def test_load_and_split_data_reproducibility() -> None:
    """Same random_state must produce identical splits."""
    split1 = load_and_split_data(random_state=42)
    split2 = load_and_split_data(random_state=42)
    np.testing.assert_array_equal(split1[0], split2[0])
    np.testing.assert_array_equal(split1[1], split2[1])


def test_load_and_split_data_different_seeds() -> None:
    """Different random states should produce different splits."""
    X_train_42, *_ = load_and_split_data(random_state=42)
    X_train_0, *_ = load_and_split_data(random_state=0)
    assert not np.array_equal(X_train_42, X_train_0)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def test_preprocess_zero_mean_unit_variance() -> None:
    """StandardScaler should produce ~zero mean and ~unit variance on training data."""
    X_train, X_test, _, _ = load_and_split_data()
    X_train_sc, X_test_sc = preprocess(X_train, X_test)
    np.testing.assert_allclose(X_train_sc.mean(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(X_train_sc.std(axis=0), 1, atol=1e-10)


def test_preprocess_shape_preserved() -> None:
    """Scaling must not change the shape of the arrays."""
    X_train, X_test, _, _ = load_and_split_data()
    X_train_sc, X_test_sc = preprocess(X_train, X_test)
    assert X_train_sc.shape == X_train.shape
    assert X_test_sc.shape == X_test.shape


def test_preprocess_no_data_leakage() -> None:
    """Test set mean should NOT be zero (scaler fitted only on train)."""
    X_train, X_test, _, _ = load_and_split_data()
    _, X_test_sc = preprocess(X_train, X_test)
    # Test set mean after train-fitted scaler should differ from 0
    assert not np.allclose(X_test_sc.mean(axis=0), 0, atol=1e-3)


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

def test_build_classifiers_returns_four_models() -> None:
    """build_classifiers should return exactly 4 models."""
    clfs = build_classifiers()
    assert len(clfs) == 4
    assert set(clfs.keys()) == {"KNN", "SVC", "LR", "RF"}


def test_evaluate_classifier_accuracy_range() -> None:
    """Accuracy must be in [0, 1]."""
    X_train, X_test, y_train, y_test = load_and_split_data()
    X_train_sc, X_test_sc = preprocess(X_train, X_test)
    clf = KNeighborsClassifier()
    result = evaluate_classifier(
        clf, X_train_sc, X_test_sc, y_train, y_test, ["malignant", "benign"]
    )
    assert 0.0 <= result["accuracy"] <= 1.0


def test_evaluate_classifier_confusion_matrix_shape() -> None:
    """Confusion matrix should be 2×2 for binary classification."""
    X_train, X_test, y_train, y_test = load_and_split_data()
    X_train_sc, X_test_sc = preprocess(X_train, X_test)
    clf = KNeighborsClassifier()
    result = evaluate_classifier(
        clf, X_train_sc, X_test_sc, y_train, y_test, ["malignant", "benign"]
    )
    assert result["confusion_matrix"].shape == (2, 2)


def test_evaluate_classifier_confusion_matrix_sum() -> None:
    """Sum of confusion matrix entries must equal number of test samples."""
    X_train, X_test, y_train, y_test = load_and_split_data()
    X_train_sc, X_test_sc = preprocess(X_train, X_test)
    clf = KNeighborsClassifier()
    result = evaluate_classifier(
        clf, X_train_sc, X_test_sc, y_train, y_test, ["malignant", "benign"]
    )
    assert result["confusion_matrix"].sum() == len(y_test)


# ---------------------------------------------------------------------------
# GridSearchCV tuning
# ---------------------------------------------------------------------------

def test_tune_classifier_returns_best_params() -> None:
    """tune_classifier should return a GridSearchCV with best_params_ set."""
    X_train, _, y_train, _ = load_and_split_data()
    X_train_sc, _ = preprocess(X_train, X_train)
    clf = KNeighborsClassifier()
    gs = tune_classifier(clf, PARAM_GRIDS["KNN"], X_train_sc, y_train, cv=3)
    assert hasattr(gs, "best_params_")
    assert "n_neighbors" in gs.best_params_
    assert gs.best_params_["n_neighbors"] in [3, 5, 7, 9]


def test_tune_classifier_best_score_in_range() -> None:
    """Best cross-validation score should be in [0, 1]."""
    X_train, _, y_train, _ = load_and_split_data()
    X_train_sc, _ = preprocess(X_train, X_train)
    clf = KNeighborsClassifier()
    gs = tune_classifier(clf, PARAM_GRIDS["KNN"], X_train_sc, y_train, cv=3)
    assert 0.0 <= gs.best_score_ <= 1.0


# ---------------------------------------------------------------------------
# Param grids
# ---------------------------------------------------------------------------

def test_param_grids_contain_all_models() -> None:
    """PARAM_GRIDS must define grids for all four classifiers."""
    assert set(PARAM_GRIDS.keys()) == {"KNN", "SVC", "LR", "RF"}


def test_param_grids_knn_values() -> None:
    """KNN param grid must match methodology specification."""
    assert PARAM_GRIDS["KNN"]["n_neighbors"] == [3, 5, 7, 9]


def test_param_grids_svc_values() -> None:
    """SVC param grid must match methodology specification."""
    assert PARAM_GRIDS["SVC"]["C"] == [0.1, 1, 10]
    assert set(PARAM_GRIDS["SVC"]["kernel"]) == {"linear", "rbf"}


# ---------------------------------------------------------------------------
# Full experiment integration test
# ---------------------------------------------------------------------------

def test_run_experiment_1_returns_dataframe() -> None:
    """run_experiment_1 should return a non-empty DataFrame."""
    df = run_experiment_1()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert "Model" in df.columns
    assert "Default Accuracy" in df.columns
    assert "Tuned Accuracy" in df.columns


def test_run_experiment_1_accuracy_above_threshold() -> None:
    """All classifiers should achieve >85% accuracy on breast cancer dataset."""
    df = run_experiment_1()
    for _, row in df.iterrows():
        assert row["Default Accuracy"] > 0.85, (
            f"{row['Model']} default accuracy {row['Default Accuracy']} < 0.85"
        )
        assert row["Tuned Accuracy"] > 0.85, (
            f"{row['Model']} tuned accuracy {row['Tuned Accuracy']} < 0.85"
        )


def test_run_experiment_1_tuned_ge_default() -> None:
    """Tuned accuracy should be >= default accuracy for all models."""
    df = run_experiment_1()
    for _, row in df.iterrows():
        assert row["Tuned Accuracy"] >= row["Default Accuracy"] - 0.02, (
            f"{row['Model']}: tuned ({row['Tuned Accuracy']}) "
            f"much worse than default ({row['Default Accuracy']})"
        )


def test_run_experiment_1_rf_svc_high_accuracy() -> None:
    """RF and SVC should achieve >95% accuracy after tuning (expected outcome)."""
    df = run_experiment_1()
    rf_row = df[df["Model"] == "RF"].iloc[0]
    svc_row = df[df["Model"] == "SVC"].iloc[0]
    assert rf_row["Tuned Accuracy"] > 0.95, (
        f"RF tuned accuracy {rf_row['Tuned Accuracy']} should be >0.95"
    )
    assert svc_row["Tuned Accuracy"] > 0.95, (
        f"SVC tuned accuracy {svc_row['Tuned Accuracy']} should be >0.95"
    )
