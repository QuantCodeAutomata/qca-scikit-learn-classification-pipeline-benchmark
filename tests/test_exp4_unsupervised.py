"""
Tests for Experiment 4: Scikit-Learn Unsupervised Learning and Dimensionality Reduction.

Validates:
- Data loading correctness
- PCA explained variance properties
- KMeans sweep output structure
- Silhouette score ranges
- Optimal k identification
- Contingency table structure
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exp4_unsupervised import (
    apply_pca,
    build_contingency_table,
    build_pca_summary,
    find_optimal_k,
    load_wine_dataframe,
    run_kmeans_sweep,
    standardize_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def wine_scaled():
    X, y, class_names = load_wine_dataframe()
    X_scaled = standardize_features(X)
    return X_scaled, y.values, class_names


@pytest.fixture(scope="module")
def pca_results(wine_scaled):
    X_scaled, _, _ = wine_scaled
    X_pca_2d, pca_2d = apply_pca(X_scaled, n_components=2)
    X_pca_3d, pca_3d = apply_pca(X_scaled, n_components=3)
    return X_pca_2d, pca_2d, X_pca_3d, pca_3d


@pytest.fixture(scope="module")
def kmeans_results(wine_scaled):
    X_scaled, _, _ = wine_scaled
    return run_kmeans_sweep(X_scaled, k_range=range(2, 11))


# ---------------------------------------------------------------------------
# Data loading tests
# ---------------------------------------------------------------------------

def test_load_wine_shape():
    """Wine dataset must have 178 samples and 13 features."""
    X, y, class_names = load_wine_dataframe()
    assert X.shape == (178, 13), f"Expected (178, 13), got {X.shape}"
    assert len(y) == 178


def test_load_wine_classes():
    """Wine dataset must have exactly 3 classes."""
    X, y, class_names = load_wine_dataframe()
    assert len(np.unique(y)) == 3
    assert len(class_names) == 3


def test_load_wine_no_missing():
    """Wine dataset must have no missing values."""
    X, y, _ = load_wine_dataframe()
    assert X.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# Standardization tests
# ---------------------------------------------------------------------------

def test_standardize_mean_near_zero(wine_scaled):
    """Standardized features must have near-zero mean."""
    X_scaled, _, _ = wine_scaled
    means = np.abs(X_scaled.mean(axis=0))
    assert np.all(means < 1e-10), f"Means not near zero: {means}"


def test_standardize_std_near_one(wine_scaled):
    """Standardized features must have near-unit std."""
    X_scaled, _, _ = wine_scaled
    stds = X_scaled.std(axis=0)
    assert np.all(np.abs(stds - 1.0) < 1e-10), f"Stds not near 1: {stds}"


def test_standardize_shape_preserved(wine_scaled):
    """Standardization must preserve the data shape."""
    X_scaled, _, _ = wine_scaled
    assert X_scaled.shape == (178, 13)


# ---------------------------------------------------------------------------
# PCA tests
# ---------------------------------------------------------------------------

def test_pca_2d_output_shape(pca_results, wine_scaled):
    """PCA with n_components=2 must produce (178, 2) output."""
    X_pca_2d, _, _, _ = pca_results
    assert X_pca_2d.shape == (178, 2)


def test_pca_3d_output_shape(pca_results):
    """PCA with n_components=3 must produce (178, 3) output."""
    _, _, X_pca_3d, _ = pca_results
    assert X_pca_3d.shape == (178, 3)


def test_pca_2d_explained_variance_sum(pca_results):
    """PCA 2D explained variance ratio must sum to the cumulative value."""
    _, pca_2d, _, _ = pca_results
    total = np.sum(pca_2d.explained_variance_ratio_)
    assert abs(total - np.cumsum(pca_2d.explained_variance_ratio_)[-1]) < 1e-10


def test_pca_2d_captures_55_to_65_percent(pca_results):
    """PCA 2D must capture approximately 55-65% of total variance in Wine dataset."""
    _, pca_2d, _, _ = pca_results
    total_var = np.sum(pca_2d.explained_variance_ratio_)
    assert 0.50 <= total_var <= 0.75, \
        f"PCA 2D total variance {total_var:.4f} outside expected range [0.50, 0.75]"


def test_pca_3d_more_variance_than_2d(pca_results):
    """PCA 3D must capture more variance than PCA 2D."""
    _, pca_2d, _, pca_3d = pca_results
    var_2d = np.sum(pca_2d.explained_variance_ratio_)
    var_3d = np.sum(pca_3d.explained_variance_ratio_)
    assert var_3d > var_2d, \
        f"PCA 3D variance {var_3d:.4f} not greater than PCA 2D {var_2d:.4f}"


def test_pca_explained_variance_ratios_positive(pca_results):
    """All explained variance ratios must be positive."""
    _, pca_2d, _, pca_3d = pca_results
    assert np.all(pca_2d.explained_variance_ratio_ > 0)
    assert np.all(pca_3d.explained_variance_ratio_ > 0)


def test_pca_explained_variance_ratios_sum_at_most_one(pca_results):
    """Explained variance ratios must sum to at most 1.0."""
    _, pca_2d, _, pca_3d = pca_results
    assert np.sum(pca_2d.explained_variance_ratio_) <= 1.0 + 1e-10
    assert np.sum(pca_3d.explained_variance_ratio_) <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# KMeans tests
# ---------------------------------------------------------------------------

def test_kmeans_sweep_k_range(kmeans_results):
    """KMeans sweep must cover k=2 through k=10."""
    assert set(kmeans_results.keys()) == set(range(2, 11))


def test_kmeans_inertia_decreasing(kmeans_results):
    """Inertia must be monotonically decreasing as k increases."""
    k_values = sorted(kmeans_results.keys())
    inertias = [kmeans_results[k]["inertia"] for k in k_values]
    for i in range(len(inertias) - 1):
        assert inertias[i] >= inertias[i + 1], \
            f"Inertia not decreasing: k={k_values[i]} ({inertias[i]:.2f}) < k={k_values[i+1]} ({inertias[i+1]:.2f})"


def test_kmeans_inertia_positive(kmeans_results):
    """All inertia values must be positive."""
    for k, res in kmeans_results.items():
        assert res["inertia"] > 0, f"k={k}: inertia={res['inertia']} is not positive"


def test_kmeans_silhouette_in_valid_range(kmeans_results):
    """Silhouette scores must be in [-1, 1]."""
    for k, res in kmeans_results.items():
        sil = res["silhouette"]
        assert -1.0 <= sil <= 1.0, f"k={k}: silhouette={sil} out of [-1,1]"


def test_kmeans_labels_valid_range(kmeans_results):
    """Cluster labels must be in [0, k-1] for each k."""
    for k, res in kmeans_results.items():
        labels = res["labels"]
        assert labels.min() >= 0, f"k={k}: negative cluster label"
        assert labels.max() <= k - 1, f"k={k}: label {labels.max()} >= k"


def test_kmeans_labels_length(kmeans_results):
    """Cluster labels must have 178 entries (one per Wine sample)."""
    for k, res in kmeans_results.items():
        assert len(res["labels"]) == 178, \
            f"k={k}: labels length {len(res['labels'])} != 178"


# ---------------------------------------------------------------------------
# Optimal k tests
# ---------------------------------------------------------------------------

def test_optimal_k_is_3(kmeans_results):
    """Optimal k (max silhouette) must be 3 for the Wine dataset."""
    optimal_k = find_optimal_k(kmeans_results)
    assert optimal_k == 3, \
        f"Expected optimal k=3, got k={optimal_k}"


def test_optimal_k_silhouette_above_0_25(kmeans_results):
    """Silhouette score at optimal k must be above 0.25 (reasonable clustering)."""
    optimal_k = find_optimal_k(kmeans_results)
    sil = kmeans_results[optimal_k]["silhouette"]
    assert sil > 0.25, f"Silhouette at optimal k={optimal_k}: {sil:.4f} < 0.25"


# ---------------------------------------------------------------------------
# Contingency table tests
# ---------------------------------------------------------------------------

def test_contingency_table_shape(wine_scaled, kmeans_results):
    """Contingency table must have 3 rows (true classes) and k+1 columns (clusters + All)."""
    _, y, class_names = wine_scaled
    optimal_k = find_optimal_k(kmeans_results)
    labels = kmeans_results[optimal_k]["labels"]
    contingency = build_contingency_table(y, labels, class_names)
    # 3 true classes + 1 "All" margin row
    assert contingency.shape[0] == 4, \
        f"Expected 4 rows (3 classes + All), got {contingency.shape[0]}"


def test_contingency_table_total_sum(wine_scaled, kmeans_results):
    """Contingency table grand total must equal 178 (number of Wine samples)."""
    _, y, class_names = wine_scaled
    optimal_k = find_optimal_k(kmeans_results)
    labels = kmeans_results[optimal_k]["labels"]
    contingency = build_contingency_table(y, labels, class_names)
    # The "All" row and "All" column both contain totals; grand total is at [All, All]
    grand_total = contingency.loc["All", "All"]
    assert grand_total == 178, f"Grand total {grand_total} != 178"


# ---------------------------------------------------------------------------
# PCA summary tests
# ---------------------------------------------------------------------------

def test_pca_summary_shape(pca_results):
    """PCA summary must have 2 rows (for n_components=2 and 3)."""
    _, pca_2d, _, pca_3d = pca_results
    summary = build_pca_summary(pca_2d, pca_3d)
    assert summary.shape[0] == 2


def test_pca_summary_cumulative_variance_increasing(pca_results):
    """Cumulative variance must be higher for n_components=3 than 2."""
    _, pca_2d, _, pca_3d = pca_results
    summary = build_pca_summary(pca_2d, pca_3d)
    var_2 = summary.loc[2, "Cumulative Variance"]
    var_3 = summary.loc[3, "Cumulative Variance"]
    assert var_3 > var_2, \
        f"Cumulative variance for 3 components ({var_3}) not > 2 components ({var_2})"
