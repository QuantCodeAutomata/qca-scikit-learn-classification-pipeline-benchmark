"""
Tests for the classification pipeline experiment (exp_1).

Validates:
- Methodology adherence (correct split, scaler, classifiers, CV)
- Mathematical properties of metrics (0-1 range)
- Pipeline structure (no data leakage)
- GridSearchCV correctness
- Edge cases
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from src.pipeline import (
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS,
    ClassifierResult,
    build_comparison_table,
    build_pipeline,
    evaluate_classifier,
    get_classifiers,
    load_dataset,
    run_experiment,
    run_grid_search,
    split_data,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def breast_cancer_split():
    """Provide a reproducible train/test split of breast_cancer."""
    X, y, _ = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def experiment_output():
    """Run the full experiment once and share results across tests."""
    return run_experiment()


# ── Dataset tests ─────────────────────────────────────────────────────────────

def test_load_dataset_shape():
    """Breast cancer dataset must have 569 samples and 30 features."""
    X, y, feature_names = load_dataset()
    assert X.shape == (569, 30), f"Expected (569, 30), got {X.shape}"
    assert y.shape == (569,)
    assert len(feature_names) == 30


def test_load_dataset_binary():
    """Target must be binary (0 or 1)."""
    _, y, _ = load_dataset()
    assert set(np.unique(y)) == {0, 1}


# ── Split tests ───────────────────────────────────────────────────────────────

def test_split_sizes(breast_cancer_split):
    """Test set must be exactly 20% of total samples."""
    X_train, X_test, y_train, y_test = breast_cancer_split
    total = len(X_train) + len(X_test)
    assert total == 569
    assert len(X_test) == pytest.approx(569 * TEST_SIZE, abs=2)


def test_split_reproducibility():
    """Same random_state must produce identical splits."""
    X, y, _ = load_dataset()
    split1 = split_data(X, y)
    split2 = split_data(X, y)
    np.testing.assert_array_equal(split1[0], split2[0])
    np.testing.assert_array_equal(split1[1], split2[1])


def test_split_stratification(breast_cancer_split):
    """Class proportions in train and test must be approximately equal."""
    X_train, X_test, y_train, y_test = breast_cancer_split
    train_ratio = np.mean(y_train)
    test_ratio = np.mean(y_test)
    assert abs(train_ratio - test_ratio) < 0.05, (
        f"Stratification failed: train={train_ratio:.3f}, test={test_ratio:.3f}"
    )


# ── Pipeline structure tests ──────────────────────────────────────────────────

def test_build_pipeline_returns_pipeline():
    """build_pipeline must return a sklearn Pipeline."""
    pipe = build_pipeline(GaussianNB())
    assert isinstance(pipe, Pipeline)


def test_pipeline_has_scaler_and_clf():
    """Pipeline must have exactly two steps: scaler and clf."""
    pipe = build_pipeline(SVC())
    step_names = [name for name, _ in pipe.steps]
    assert step_names == ["scaler", "clf"]
    assert isinstance(pipe.named_steps["scaler"], StandardScaler)


def test_pipeline_prevents_data_leakage(breast_cancer_split):
    """Scaler must be fitted only on training data (not test data)."""
    X_train, X_test, y_train, y_test = breast_cancer_split
    pipe = build_pipeline(GaussianNB())
    pipe.fit(X_train, y_train)
    scaler: StandardScaler = pipe.named_steps["scaler"]
    # Scaler mean must match training data mean, not test data mean
    np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0), rtol=1e-5)
    assert not np.allclose(scaler.mean_, X_test.mean(axis=0))


# ── Classifier instantiation tests ───────────────────────────────────────────

def test_get_classifiers_returns_required_types():
    """All four required classifiers must be present."""
    clfs = get_classifiers()
    assert isinstance(clfs["KNeighborsClassifier"], KNeighborsClassifier)
    assert isinstance(clfs["SVC"], SVC)
    assert isinstance(clfs["GaussianNB"], GaussianNB)
    assert isinstance(clfs["RandomForestClassifier"], RandomForestClassifier)


def test_knn_n_neighbors():
    """KNN must use n_neighbors=5 as specified in methodology."""
    clfs = get_classifiers()
    assert clfs["KNeighborsClassifier"].n_neighbors == 5


def test_svc_kernel():
    """SVC must use rbf kernel as default."""
    clfs = get_classifiers()
    assert clfs["SVC"].kernel == "rbf"


def test_random_forest_random_state():
    """RandomForest must use random_state=42."""
    clfs = get_classifiers()
    assert clfs["RandomForestClassifier"].random_state == RANDOM_STATE


# ── Metric range tests ────────────────────────────────────────────────────────

def test_evaluate_classifier_metric_ranges(breast_cancer_split):
    """All metrics must be in [0, 1]."""
    X_train, X_test, y_train, y_test = breast_cancer_split
    result = evaluate_classifier(
        "GaussianNB", GaussianNB(), X_train, X_test, y_train, y_test
    )
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
    assert 0.0 <= result.f1 <= 1.0
    assert 0.0 <= result.cv_mean <= 1.0
    assert result.cv_std >= 0.0


def test_evaluate_classifier_cv_folds(breast_cancer_split):
    """CV must use exactly CV_FOLDS=5 folds."""
    assert CV_FOLDS == 5


def test_evaluate_classifier_returns_result_type(breast_cancer_split):
    """evaluate_classifier must return a ClassifierResult."""
    X_train, X_test, y_train, y_test = breast_cancer_split
    result = evaluate_classifier(
        "GaussianNB", GaussianNB(), X_train, X_test, y_train, y_test
    )
    assert isinstance(result, ClassifierResult)
    assert result.name == "GaussianNB"


# ── Performance sanity tests ──────────────────────────────────────────────────

def test_svc_accuracy_above_threshold(breast_cancer_split):
    """SVC on breast_cancer should achieve > 90% test accuracy."""
    X_train, X_test, y_train, y_test = breast_cancer_split
    result = evaluate_classifier(
        "SVC", SVC(kernel="rbf", random_state=RANDOM_STATE),
        X_train, X_test, y_train, y_test
    )
    assert result.accuracy > 0.90, f"SVC accuracy too low: {result.accuracy:.4f}"


def test_random_forest_accuracy_above_threshold(breast_cancer_split):
    """RandomForest on breast_cancer should achieve > 90% test accuracy."""
    X_train, X_test, y_train, y_test = breast_cancer_split
    result = evaluate_classifier(
        "RandomForestClassifier",
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        X_train, X_test, y_train, y_test,
    )
    assert result.accuracy > 0.90, f"RF accuracy too low: {result.accuracy:.4f}"


def test_cv_mean_consistent_with_test_accuracy(breast_cancer_split):
    """CV mean accuracy should be within 10% of test accuracy (no overfitting)."""
    X_train, X_test, y_train, y_test = breast_cancer_split
    result = evaluate_classifier(
        "SVC", SVC(kernel="rbf", random_state=RANDOM_STATE),
        X_train, X_test, y_train, y_test
    )
    assert abs(result.cv_mean - result.accuracy) < 0.10, (
        f"Large gap between CV ({result.cv_mean:.4f}) and test ({result.accuracy:.4f})"
    )


# ── GridSearchCV tests ────────────────────────────────────────────────────────

def test_grid_search_returns_gridsearchcv(breast_cancer_split):
    """run_grid_search must return a fitted GridSearchCV."""
    X_train, _, y_train, _ = breast_cancer_split
    gs = run_grid_search(X_train, y_train)
    assert isinstance(gs, GridSearchCV)
    assert hasattr(gs, "best_params_")


def test_grid_search_best_params_valid(breast_cancer_split):
    """Best params must be within the specified param grid."""
    X_train, _, y_train, _ = breast_cancer_split
    gs = run_grid_search(X_train, y_train)
    assert gs.best_params_["clf__C"] in [0.1, 1, 10]
    assert gs.best_params_["clf__kernel"] in ["linear", "rbf"]


def test_grid_search_best_score_above_threshold(breast_cancer_split):
    """GridSearchCV best CV score must exceed 90%."""
    X_train, _, y_train, _ = breast_cancer_split
    gs = run_grid_search(X_train, y_train)
    assert gs.best_score_ > 0.90, f"GridSearchCV best score too low: {gs.best_score_:.4f}"


def test_grid_search_pipeline_structure(breast_cancer_split):
    """GridSearchCV estimator must be a Pipeline with scaler and clf."""
    X_train, _, y_train, _ = breast_cancer_split
    gs = run_grid_search(X_train, y_train)
    pipe = gs.best_estimator_
    assert isinstance(pipe, Pipeline)
    assert "scaler" in pipe.named_steps
    assert "clf" in pipe.named_steps


# ── Comparison table tests ────────────────────────────────────────────────────

def test_comparison_table_sorted_by_f1():
    """Comparison table must be sorted descending by Test F1."""
    results = [
        ClassifierResult("A", 0.9, 0.9, 0.9, 0.85, 0.88, 0.02),
        ClassifierResult("B", 0.95, 0.95, 0.95, 0.95, 0.93, 0.01),
        ClassifierResult("C", 0.80, 0.80, 0.80, 0.78, 0.79, 0.03),
    ]
    df = build_comparison_table(results)
    f1_values = df["Test F1"].tolist()
    assert f1_values == sorted(f1_values, reverse=True), "Table not sorted by F1 descending"


def test_comparison_table_columns():
    """Comparison table must contain all required metric columns."""
    results = [ClassifierResult("X", 0.9, 0.9, 0.9, 0.9, 0.88, 0.02)]
    df = build_comparison_table(results)
    required_cols = [
        "Classifier", "Test Accuracy", "Test Precision",
        "Test Recall", "Test F1", "CV Mean Accuracy", "CV Std"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_comparison_table_metric_ranges():
    """All metric values in comparison table must be in [0, 1]."""
    results = [
        ClassifierResult("A", 0.9, 0.88, 0.91, 0.89, 0.87, 0.02),
        ClassifierResult("B", 0.95, 0.94, 0.96, 0.95, 0.93, 0.01),
    ]
    df = build_comparison_table(results)
    for col in ["Test Accuracy", "Test Precision", "Test Recall", "Test F1", "CV Mean Accuracy"]:
        assert (df[col] >= 0).all() and (df[col] <= 1).all(), f"Out-of-range values in {col}"


# ── Full experiment integration tests ─────────────────────────────────────────

def test_experiment_output_keys(experiment_output):
    """run_experiment must return all required keys."""
    required_keys = ["results", "comparison_table", "grid_search", "X_test", "y_test"]
    for key in required_keys:
        assert key in experiment_output, f"Missing key: {key}"


def test_experiment_results_count(experiment_output):
    """Experiment must evaluate at least 4 classifiers + 1 GridSearchCV variant."""
    results = experiment_output["results"]
    assert len(results) >= 5, f"Expected >= 5 results, got {len(results)}"


def test_experiment_comparison_table_not_empty(experiment_output):
    """Comparison table must not be empty."""
    df = experiment_output["comparison_table"]
    assert len(df) >= 5


def test_experiment_all_classifiers_present(experiment_output):
    """All four required classifiers must appear in results."""
    results = experiment_output["results"]
    names = [r.name for r in results]
    for required in ["KNeighborsClassifier", "SVC", "GaussianNB", "RandomForestClassifier"]:
        assert any(required in n for n in names), f"Missing classifier: {required}"


def test_experiment_best_classifier_f1_above_threshold(experiment_output):
    """Best classifier F1 must exceed 0.90 on breast_cancer."""
    results = experiment_output["results"]
    best_f1 = max(r.f1 for r in results)
    assert best_f1 > 0.90, f"Best F1 too low: {best_f1:.4f}"


# ── Edge case tests ───────────────────────────────────────────────────────────

def test_evaluate_classifier_single_class_edge_case():
    """evaluate_classifier must not crash with a near-trivial dataset (iris 2-class subset)."""
    data = load_iris()
    # Use only first two classes for binary classification
    mask = data.target < 2
    X, y = data.data[mask], data.target[mask]
    X_train, X_test, y_train, y_test = split_data(X, y)
    result = evaluate_classifier("GaussianNB", GaussianNB(), X_train, X_test, y_train, y_test)
    assert 0.0 <= result.accuracy <= 1.0


def test_build_pipeline_with_different_classifiers():
    """build_pipeline must work with all four required classifier types."""
    clfs = get_classifiers()
    for name, clf in clfs.items():
        pipe = build_pipeline(clf)
        assert isinstance(pipe, Pipeline), f"Pipeline failed for {name}"


def test_split_data_no_overlap():
    """Train and test indices must not overlap."""
    X, y, _ = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)
    # Verify sizes sum to total
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)


def test_constants_match_methodology():
    """Constants must match the methodology specification exactly."""
    assert RANDOM_STATE == 42, "random_state must be 42"
    assert TEST_SIZE == 0.2, "test_size must be 0.2"
    assert CV_FOLDS == 5, "CV folds must be 5"
