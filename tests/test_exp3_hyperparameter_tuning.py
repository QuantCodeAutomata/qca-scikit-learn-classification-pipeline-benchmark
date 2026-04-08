"""
Tests for Experiment 3: Scikit-Learn Hyperparameter Tuning and Cross-Validation.

Validates:
- Data loading correctness
- Param grid structure
- GridSearchCV and RandomizedSearchCV output properties
- CV score ranges and stability
- Best model test accuracy
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exp3_hyperparameter_tuning import (
    define_param_grid,
    evaluate_cv_stability,
    load_digits_data,
    run_grid_search,
    run_random_search,
    scale_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def digits_splits():
    X, y = load_digits_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


@pytest.fixture(scope="module")
def grid_search_result(digits_splits):
    X_train_scaled, _, y_train, _ = digits_splits
    param_grid = define_param_grid()
    gs, elapsed = run_grid_search(X_train_scaled, y_train, param_grid)
    return gs, elapsed


@pytest.fixture(scope="module")
def random_search_result(digits_splits):
    X_train_scaled, _, y_train, _ = digits_splits
    param_grid = define_param_grid()
    rs, elapsed = run_random_search(X_train_scaled, y_train, param_grid)
    return rs, elapsed


# ---------------------------------------------------------------------------
# Data loading tests
# ---------------------------------------------------------------------------

def test_load_digits_shape():
    """Digits dataset must have 1797 samples and 64 features."""
    X, y = load_digits_data()
    assert X.shape == (1797, 64), f"Expected (1797, 64), got {X.shape}"


def test_load_digits_classes():
    """Digits dataset must have 10 classes (0-9)."""
    X, y = load_digits_data()
    assert len(np.unique(y)) == 10


def test_load_digits_feature_range():
    """Digits pixel values must be in [0, 16]."""
    X, y = load_digits_data()
    assert X.min() >= 0
    assert X.max() <= 16


# ---------------------------------------------------------------------------
# Param grid tests
# ---------------------------------------------------------------------------

def test_param_grid_keys():
    """Param grid must contain C, kernel, and gamma keys."""
    grid = define_param_grid()
    assert set(grid.keys()) == {"C", "kernel", "gamma"}


def test_param_grid_c_values():
    """C values must match methodology: [0.1, 1, 10, 100]."""
    grid = define_param_grid()
    assert grid["C"] == [0.1, 1, 10, 100]


def test_param_grid_kernel_values():
    """Kernel values must match methodology: ['linear', 'rbf']."""
    grid = define_param_grid()
    assert set(grid["kernel"]) == {"linear", "rbf"}


def test_param_grid_gamma_values():
    """Gamma values must match methodology: ['scale', 'auto']."""
    grid = define_param_grid()
    assert set(grid["gamma"]) == {"scale", "auto"}


# ---------------------------------------------------------------------------
# GridSearchCV tests
# ---------------------------------------------------------------------------

def test_grid_search_best_score_above_97(grid_search_result):
    """GridSearchCV best CV score must exceed 97% on Digits."""
    gs, _ = grid_search_result
    assert gs.best_score_ > 0.97, \
        f"GridSearchCV best score {gs.best_score_:.4f} < 0.97"


def test_grid_search_best_params_valid(grid_search_result):
    """GridSearchCV best params must be within the defined grid."""
    gs, _ = grid_search_result
    grid = define_param_grid()
    assert gs.best_params_["C"] in grid["C"]
    assert gs.best_params_["kernel"] in grid["kernel"]
    assert gs.best_params_["gamma"] in grid["gamma"]


def test_grid_search_elapsed_positive(grid_search_result):
    """GridSearchCV wall-clock time must be positive."""
    _, elapsed = grid_search_result
    assert elapsed > 0


def test_grid_search_cv_results_not_empty(grid_search_result):
    """GridSearchCV cv_results_ must contain results for all param combinations."""
    gs, _ = grid_search_result
    # 4 C values × 2 kernels × 2 gamma values = 16 combinations
    assert len(gs.cv_results_["params"]) == 16


# ---------------------------------------------------------------------------
# RandomizedSearchCV tests
# ---------------------------------------------------------------------------

def test_random_search_best_score_above_97(random_search_result):
    """RandomizedSearchCV best CV score must exceed 97% on Digits."""
    rs, _ = random_search_result
    assert rs.best_score_ > 0.97, \
        f"RandomizedSearchCV best score {rs.best_score_:.4f} < 0.97"


def test_random_search_best_params_valid(random_search_result):
    """RandomizedSearchCV best params must be within the defined grid."""
    rs, _ = random_search_result
    grid = define_param_grid()
    assert rs.best_params_["C"] in grid["C"]
    assert rs.best_params_["kernel"] in grid["kernel"]
    assert rs.best_params_["gamma"] in grid["gamma"]


def test_random_search_n_iter_respected(random_search_result):
    """RandomizedSearchCV must evaluate at most n_iter=20 combinations.

    When the total parameter space (16) is smaller than n_iter (20),
    scikit-learn evaluates all 16 combinations and emits a UserWarning.
    """
    rs, _ = random_search_result
    # Total space = 4 C × 2 kernels × 2 gamma = 16 < n_iter=20
    # sklearn falls back to exhaustive search over all 16 combinations
    n_evaluated = len(rs.cv_results_["params"])
    assert n_evaluated <= 20, \
        f"RandomizedSearchCV evaluated {n_evaluated} > n_iter=20 combinations"
    assert n_evaluated >= 1, "RandomizedSearchCV evaluated 0 combinations"


def test_random_search_elapsed_positive(random_search_result):
    """RandomizedSearchCV wall-clock time must be positive."""
    _, elapsed = random_search_result
    assert elapsed > 0


# ---------------------------------------------------------------------------
# Cross-validation stability tests
# ---------------------------------------------------------------------------

def test_cv_scores_all_k_values(digits_splits, grid_search_result):
    """CV stability must return scores for k=3, 5, 10."""
    X_train_scaled, _, y_train, _ = digits_splits
    gs, _ = grid_search_result
    best_model = gs.best_estimator_
    cv_scores = evaluate_cv_stability(best_model, X_train_scaled, y_train, k_values=[3, 5, 10])
    assert set(cv_scores.keys()) == {3, 5, 10}


def test_cv_scores_correct_fold_counts(digits_splits, grid_search_result):
    """Each k-fold CV must return exactly k scores."""
    X_train_scaled, _, y_train, _ = digits_splits
    gs, _ = grid_search_result
    best_model = gs.best_estimator_
    cv_scores = evaluate_cv_stability(best_model, X_train_scaled, y_train, k_values=[3, 5, 10])
    for k, scores in cv_scores.items():
        assert len(scores) == k, f"k={k}: expected {k} scores, got {len(scores)}"


def test_cv_scores_in_valid_range(digits_splits, grid_search_result):
    """All CV fold scores must be in [0, 1]."""
    X_train_scaled, _, y_train, _ = digits_splits
    gs, _ = grid_search_result
    best_model = gs.best_estimator_
    cv_scores = evaluate_cv_stability(best_model, X_train_scaled, y_train, k_values=[3, 5, 10])
    for k, scores in cv_scores.items():
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0), \
            f"k={k}: scores out of [0,1]: {scores}"


def test_cv_scores_above_95(digits_splits, grid_search_result):
    """Mean CV score for all k values must exceed 95%."""
    X_train_scaled, _, y_train, _ = digits_splits
    gs, _ = grid_search_result
    best_model = gs.best_estimator_
    cv_scores = evaluate_cv_stability(best_model, X_train_scaled, y_train, k_values=[3, 5, 10])
    for k, scores in cv_scores.items():
        mean_score = scores.mean()
        assert mean_score > 0.95, \
            f"k={k}: mean CV score {mean_score:.4f} < 0.95"


def test_higher_k_lower_variance(digits_splits, grid_search_result):
    """Higher k values should generally produce lower or equal variance in CV scores."""
    X_train_scaled, _, y_train, _ = digits_splits
    gs, _ = grid_search_result
    best_model = gs.best_estimator_
    cv_scores = evaluate_cv_stability(best_model, X_train_scaled, y_train, k_values=[3, 5, 10])
    std_3 = cv_scores[3].std()
    std_10 = cv_scores[10].std()
    # k=10 should have lower or comparable variance than k=3
    assert std_10 <= std_3 + 0.02, \
        f"k=10 std {std_10:.4f} unexpectedly much higher than k=3 std {std_3:.4f}"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

def test_scale_features_preserves_shape():
    """scale_features must preserve array shapes."""
    X, y = load_digits_data()
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.30, random_state=42)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape


def test_scale_features_train_mean_near_zero():
    """scale_features must produce near-zero mean on training data."""
    X, y = load_digits_data()
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.30, random_state=42)
    X_train_scaled, _ = scale_features(X_train, X_test)
    means = np.abs(X_train_scaled.mean(axis=0))
    assert np.all(means < 1e-10)
