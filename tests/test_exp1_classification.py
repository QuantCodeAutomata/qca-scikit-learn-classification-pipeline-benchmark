"""
Tests for Experiment 1: Scikit-Learn Classification Pipeline Benchmark.

Validates:
- Data loading and EDA correctness
- Preprocessing (StandardScaler) properties
- Classifier training and prediction shapes
- Metric ranges and mathematical properties
- Confusion matrix structure
- Comparison table structure
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exp1_classification import (
    build_classifiers,
    build_comparison_table,
    exploratory_data_analysis,
    load_iris_dataframe,
    preprocess_features,
    train_and_evaluate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def iris_data():
    X, y = load_iris_dataframe()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_train_scaled, X_test_scaled = preprocess_features(X_train, X_test)
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values


@pytest.fixture(scope="module")
def eval_results(iris_data):
    X_train_scaled, X_test_scaled, y_train, y_test = iris_data
    classifiers = build_classifiers()
    return train_and_evaluate(classifiers, X_train_scaled, X_test_scaled, y_train, y_test)


# ---------------------------------------------------------------------------
# Data loading tests
# ---------------------------------------------------------------------------

def test_load_iris_dataframe_shape():
    """Iris dataset must have 150 samples and 4 features."""
    X, y = load_iris_dataframe()
    assert X.shape == (150, 4), f"Expected (150, 4), got {X.shape}"
    assert len(y) == 150


def test_load_iris_no_missing_values():
    """Iris dataset must have no missing values."""
    X, y = load_iris_dataframe()
    assert X.isnull().sum().sum() == 0


def test_load_iris_target_classes():
    """Iris target must have exactly 3 unique classes."""
    _, y = load_iris_dataframe()
    assert len(y.unique()) == 3


def test_load_iris_class_balance():
    """Each class in Iris must have exactly 50 samples."""
    _, y = load_iris_dataframe()
    counts = y.value_counts()
    assert all(counts == 50), f"Expected balanced classes, got {counts.to_dict()}"


# ---------------------------------------------------------------------------
# EDA tests
# ---------------------------------------------------------------------------

def test_eda_returns_correct_keys():
    """EDA function must return all required keys."""
    X, y = load_iris_dataframe()
    eda = exploratory_data_analysis(X, y)
    required_keys = {"shape", "dtypes", "missing_values", "class_distribution", "describe"}
    assert required_keys.issubset(set(eda.keys()))


def test_eda_shape_matches():
    """EDA shape must match actual data shape."""
    X, y = load_iris_dataframe()
    eda = exploratory_data_analysis(X, y)
    assert eda["shape"] == (150, 4)


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

def test_scaled_train_mean_near_zero():
    """StandardScaler must produce near-zero mean on training data."""
    X, y = load_iris_dataframe()
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.30, random_state=42)
    X_train_scaled, _ = preprocess_features(X_train, X_test)
    means = np.abs(X_train_scaled.mean(axis=0))
    assert np.all(means < 1e-10), f"Training means not near zero: {means}"


def test_scaled_train_std_near_one():
    """StandardScaler must produce near-unit std on training data."""
    X, y = load_iris_dataframe()
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.30, random_state=42)
    X_train_scaled, _ = preprocess_features(X_train, X_test)
    stds = X_train_scaled.std(axis=0)
    assert np.all(np.abs(stds - 1.0) < 1e-10), f"Training stds not near 1: {stds}"


def test_scaled_shapes_preserved():
    """Scaling must preserve the number of samples and features."""
    X, y = load_iris_dataframe()
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.30, random_state=42)
    X_train_scaled, X_test_scaled = preprocess_features(X_train, X_test)
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape


# ---------------------------------------------------------------------------
# Classifier tests
# ---------------------------------------------------------------------------

def test_build_classifiers_returns_three():
    """build_classifiers must return exactly 3 classifiers."""
    classifiers = build_classifiers()
    assert len(classifiers) == 3


def test_build_classifiers_names():
    """Classifier names must match the methodology specification."""
    classifiers = build_classifiers()
    expected = {"LogisticRegression", "KNeighborsClassifier", "SVC"}
    assert set(classifiers.keys()) == expected


def test_predictions_shape(eval_results, iris_data):
    """Predictions must have the same length as the test set."""
    _, _, _, y_test = iris_data
    for name, res in eval_results.items():
        assert len(res["y_pred"]) == len(y_test), \
            f"{name}: prediction length mismatch"


def test_predictions_valid_classes(eval_results):
    """All predictions must be valid class labels (0, 1, or 2)."""
    valid_classes = {0, 1, 2}
    for name, res in eval_results.items():
        pred_classes = set(np.unique(res["y_pred"]))
        assert pred_classes.issubset(valid_classes), \
            f"{name}: invalid predicted classes {pred_classes}"


# ---------------------------------------------------------------------------
# Metric tests
# ---------------------------------------------------------------------------

def test_accuracy_above_90_percent(eval_results):
    """All classifiers must achieve > 90% accuracy on Iris test set."""
    for name, res in eval_results.items():
        assert res["accuracy"] > 0.90, \
            f"{name}: accuracy {res['accuracy']:.4f} is below 90%"


def test_accuracy_in_valid_range(eval_results):
    """Accuracy must be between 0 and 1."""
    for name, res in eval_results.items():
        assert 0.0 <= res["accuracy"] <= 1.0, \
            f"{name}: accuracy {res['accuracy']} out of [0,1]"


def test_confusion_matrix_shape(eval_results):
    """Confusion matrix must be 3×3 for the 3-class Iris problem."""
    for name, res in eval_results.items():
        cm = res["confusion_matrix"]
        assert cm.shape == (3, 3), f"{name}: CM shape {cm.shape} != (3,3)"


def test_confusion_matrix_sum_equals_test_size(eval_results, iris_data):
    """Sum of confusion matrix must equal the number of test samples."""
    _, _, _, y_test = iris_data
    for name, res in eval_results.items():
        cm_sum = res["confusion_matrix"].sum()
        assert cm_sum == len(y_test), \
            f"{name}: CM sum {cm_sum} != test size {len(y_test)}"


def test_confusion_matrix_non_negative(eval_results):
    """All confusion matrix entries must be non-negative."""
    for name, res in eval_results.items():
        assert np.all(res["confusion_matrix"] >= 0), \
            f"{name}: negative CM entries"


def test_report_contains_required_metrics(eval_results):
    """Classification report must contain precision, recall, f1-score."""
    for name, res in eval_results.items():
        report = res["report"]
        assert "macro avg" in report, f"{name}: missing 'macro avg'"
        for metric in ["precision", "recall", "f1-score"]:
            assert metric in report["macro avg"], \
                f"{name}: missing '{metric}' in macro avg"


def test_precision_recall_f1_in_valid_range(eval_results):
    """Precision, recall, and F1 must be in [0, 1]."""
    for name, res in eval_results.items():
        macro = res["report"]["macro avg"]
        for metric in ["precision", "recall", "f1-score"]:
            val = macro[metric]
            assert 0.0 <= val <= 1.0, \
                f"{name}: {metric}={val} out of [0,1]"


# ---------------------------------------------------------------------------
# Comparison table tests
# ---------------------------------------------------------------------------

def test_comparison_table_shape(eval_results):
    """Comparison table must have 3 rows (one per classifier)."""
    table = build_comparison_table(eval_results)
    assert table.shape[0] == 3


def test_comparison_table_columns(eval_results):
    """Comparison table must contain all required metric columns."""
    table = build_comparison_table(eval_results)
    required_cols = {"Accuracy", "Precision (macro)", "Recall (macro)", "F1-Score (macro)"}
    assert required_cols.issubset(set(table.columns))


def test_comparison_table_accuracy_values(eval_results):
    """Comparison table accuracy values must match direct computation."""
    table = build_comparison_table(eval_results)
    for name, res in eval_results.items():
        expected = round(res["accuracy"], 4)
        actual = table.loc[name, "Accuracy"]
        assert abs(actual - expected) < 1e-6, \
            f"{name}: table accuracy {actual} != computed {expected}"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

def test_single_sample_prediction():
    """Classifiers must handle single-sample prediction without error."""
    X, y = load_iris_dataframe()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_train_scaled, X_test_scaled = preprocess_features(X_train, X_test)
    classifiers = build_classifiers()
    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train.values)
        pred = clf.predict(X_test_scaled[:1])
        assert len(pred) == 1, f"{name}: single-sample prediction failed"


def test_reproducibility_with_fixed_random_state():
    """Results must be identical across two runs with the same random_state."""
    X, y = load_iris_dataframe()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_train_scaled, X_test_scaled = preprocess_features(X_train, X_test)
    classifiers1 = build_classifiers()
    classifiers2 = build_classifiers()
    for name in classifiers1:
        classifiers1[name].fit(X_train_scaled, y_train.values)
        classifiers2[name].fit(X_train_scaled, y_train.values)
        pred1 = classifiers1[name].predict(X_test_scaled)
        pred2 = classifiers2[name].predict(X_test_scaled)
        assert np.array_equal(pred1, pred2), \
            f"{name}: predictions differ across runs"
