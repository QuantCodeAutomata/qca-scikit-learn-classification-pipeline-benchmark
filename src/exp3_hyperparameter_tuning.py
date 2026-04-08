"""
Experiment 3: Scikit-Learn Hyperparameter Tuning and Cross-Validation Study
============================================================================
Compares GridSearchCV and RandomizedSearchCV for SVC hyperparameter optimization
on the Digits dataset, and evaluates cross-validation stability across k=3,5,10.

Methodology (DataCamp Scikit-Learn Cheat Sheet):
  1. Load Digits dataset, split 70/30 (random_state=42)
  2. Apply StandardScaler
  3. Define SVC base estimator
  4. Param grid: C=[0.1,1,10,100], kernel=['linear','rbf'], gamma=['scale','auto']
  5. GridSearchCV (cv=5, scoring='accuracy', n_jobs=-1) → best params, score, time
  6. RandomizedSearchCV (n_iter=20, cv=5, random_state=42, n_jobs=-1) → same
  7. cross_val_score with cv=3,5,10 on best model → mean ± std
  8. Evaluate best model on held-out test set
  9. Box plots of CV score distributions for each k

# Using scikit-learn GridSearchCV, RandomizedSearchCV, cross_val_score — Context7 confirmed
# Using matplotlib for box plots — Context7 confirmed
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_digits_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load the Digits dataset.

    Returns
    -------
    X : np.ndarray  shape (1797, 64)
    y : np.ndarray  shape (1797,)
    """
    digits = load_digits()
    print(f"Digits dataset: {digits.data.shape[0]} samples, "
          f"{digits.data.shape[1]} features, {len(np.unique(digits.target))} classes")
    return digits.data, digits.target


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def scale_features(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit StandardScaler on training data and transform both splits.

    Parameters
    ----------
    X_train, X_test : np.ndarray

    Returns
    -------
    X_train_scaled, X_test_scaled : np.ndarray
    """
    # Using scikit-learn StandardScaler — Context7 confirmed
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------

def define_param_grid() -> Dict:
    """Return the SVC hyperparameter grid as specified in the methodology.

    Returns
    -------
    dict
        C, kernel, gamma search space.
    """
    return {
        "C": [0.1, 1, 10, 100],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    }


def run_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict,
) -> Tuple[GridSearchCV, float]:
    """Run GridSearchCV on SVC and measure wall-clock time.

    Parameters
    ----------
    X_train : np.ndarray
    y_train : np.ndarray
    param_grid : dict

    Returns
    -------
    grid_search : GridSearchCV (fitted)
    elapsed : float  Wall-clock time in seconds.
    """
    # Using scikit-learn GridSearchCV — Context7 confirmed
    estimator = SVC()
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    t0 = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"\n=== GridSearchCV ===")
    print(f"Best params:  {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    print(f"Wall-clock time: {elapsed:.2f}s")
    return grid_search, elapsed


def run_random_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict,
) -> Tuple[RandomizedSearchCV, float]:
    """Run RandomizedSearchCV on SVC and measure wall-clock time.

    Parameters
    ----------
    X_train : np.ndarray
    y_train : np.ndarray
    param_grid : dict

    Returns
    -------
    random_search : RandomizedSearchCV (fitted)
    elapsed : float  Wall-clock time in seconds.
    """
    # Using scikit-learn RandomizedSearchCV — Context7 confirmed
    estimator = SVC()
    random_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=20,
        cv=5,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
    )
    t0 = time.time()
    random_search.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"\n=== RandomizedSearchCV ===")
    print(f"Best params:  {random_search.best_params_}")
    print(f"Best CV score: {random_search.best_score_:.4f}")
    print(f"Wall-clock time: {elapsed:.2f}s")
    return random_search, elapsed


# ---------------------------------------------------------------------------
# Cross-validation stability
# ---------------------------------------------------------------------------

def evaluate_cv_stability(
    best_model: SVC,
    X_train: np.ndarray,
    y_train: np.ndarray,
    k_values: List[int] = None,
) -> Dict[int, np.ndarray]:
    """Run cross_val_score for multiple k values on the best model.

    Parameters
    ----------
    best_model : SVC
        Best estimator from search.
    X_train : np.ndarray
    y_train : np.ndarray
    k_values : list of int
        k values for k-fold CV (default: [3, 5, 10]).

    Returns
    -------
    cv_scores : dict
        k → array of fold scores.
    """
    if k_values is None:
        k_values = [3, 5, 10]

    cv_scores: Dict[int, np.ndarray] = {}
    print("\n=== Cross-Validation Stability ===")
    for k in k_values:
        # Using scikit-learn cross_val_score — Context7 confirmed
        scores = cross_val_score(best_model, X_train, y_train, cv=k, scoring="accuracy")
        cv_scores[k] = scores
        print(f"  k={k:2d}: mean={scores.mean():.4f}, std={scores.std():.4f}, "
              f"scores={np.round(scores, 4)}")
    return cv_scores


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_cv_score_distributions(
    cv_scores: Dict[int, np.ndarray],
    output_dir: str = "results",
) -> None:
    """Plot box plots of CV score distributions for each k value.

    Parameters
    ----------
    cv_scores : dict
        k → array of fold scores.
    output_dir : str
        Directory to save the figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    k_values = sorted(cv_scores.keys())
    data = [cv_scores[k] for k in k_values]
    labels = [f"k={k}" for k in k_values]

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title("Exp 3 – CV Score Distributions by k (Best SVC)", fontsize=13)
    ax.set_xlabel("k-Fold Value")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.90, 1.01)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(output_dir, "exp3_cv_score_distributions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved CV score distribution plot → {path}")


def plot_search_comparison(
    grid_score: float,
    grid_time: float,
    random_score: float,
    random_time: float,
    output_dir: str = "results",
) -> None:
    """Bar chart comparing GridSearch vs RandomSearch score and time.

    Parameters
    ----------
    grid_score, grid_time : float
    random_score, random_time : float
    output_dir : str
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    methods = ["GridSearchCV", "RandomizedSearchCV"]
    scores = [grid_score, random_score]
    times = [grid_time, random_time]

    ax1.bar(methods, scores, color=["#4C72B0", "#DD8452"], alpha=0.8, edgecolor="black")
    ax1.set_title("Best CV Accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0.95, 1.0)
    for i, v in enumerate(scores):
        ax1.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=11)

    ax2.bar(methods, times, color=["#4C72B0", "#DD8452"], alpha=0.8, edgecolor="black")
    ax2.set_title("Wall-Clock Time (seconds)")
    ax2.set_ylabel("Time (s)")
    for i, v in enumerate(times):
        ax2.text(i, v + 0.1, f"{v:.2f}s", ha="center", fontsize=11)

    plt.suptitle("Exp 3 – GridSearch vs RandomSearch Comparison", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "exp3_search_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved search comparison plot → {path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_search_summary(
    grid_search: GridSearchCV,
    grid_time: float,
    random_search: RandomizedSearchCV,
    random_time: float,
) -> pd.DataFrame:
    """Build a summary DataFrame comparing the two search strategies.

    Parameters
    ----------
    grid_search : GridSearchCV (fitted)
    grid_time : float
    random_search : RandomizedSearchCV (fitted)
    random_time : float

    Returns
    -------
    pd.DataFrame
    """
    rows = [
        {
            "Method": "GridSearchCV",
            "Best CV Score": round(grid_search.best_score_, 4),
            "Best Params": str(grid_search.best_params_),
            "Wall-Clock Time (s)": round(grid_time, 2),
        },
        {
            "Method": "RandomizedSearchCV",
            "Best CV Score": round(random_search.best_score_, 4),
            "Best Params": str(random_search.best_params_),
            "Wall-Clock Time (s)": round(random_time, 2),
        },
    ]
    df = pd.DataFrame(rows).set_index("Method")
    print("\n=== Search Strategy Comparison ===")
    print(df.to_string())
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_experiment(output_dir: str = "results") -> Dict:
    """Execute the full Exp 3 hyperparameter tuning study.

    Parameters
    ----------
    output_dir : str
        Directory for saving plots and artefacts.

    Returns
    -------
    dict
        Contains search results, CV scores, and comparison tables.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1 – Load data
    X, y = load_digits_data()

    # Step 2 – Train/test split (70/30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Step 3 – Preprocessing
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Step 4 – Define param grid
    param_grid = define_param_grid()

    # Step 5 – GridSearchCV
    grid_search, grid_time = run_grid_search(X_train_scaled, y_train, param_grid)

    # Step 6 – RandomizedSearchCV
    random_search, random_time = run_random_search(X_train_scaled, y_train, param_grid)

    # Step 7 – CV stability with best model (use GridSearch best estimator)
    best_model = grid_search.best_estimator_
    cv_scores = evaluate_cv_stability(best_model, X_train_scaled, y_train, k_values=[3, 5, 10])

    # Step 8 – Evaluate best model on held-out test set
    y_pred = best_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\n=== Best Model Test Set Evaluation ===")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    # Step 9 – Visualizations
    plot_cv_score_distributions(cv_scores, output_dir)
    plot_search_comparison(
        grid_search.best_score_, grid_time,
        random_search.best_score_, random_time,
        output_dir,
    )

    # Summary tables
    search_summary = build_search_summary(grid_search, grid_time, random_search, random_time)
    search_summary.to_csv(os.path.join(output_dir, "exp3_search_summary.csv"))

    cv_summary_rows = [
        {"k": k, "Mean Accuracy": round(v.mean(), 4), "Std": round(v.std(), 4)}
        for k, v in cv_scores.items()
    ]
    cv_summary = pd.DataFrame(cv_summary_rows).set_index("k")
    cv_summary.to_csv(os.path.join(output_dir, "exp3_cv_summary.csv"))

    return {
        "grid_search": grid_search,
        "random_search": random_search,
        "cv_scores": cv_scores,
        "test_accuracy": test_accuracy,
        "search_summary": search_summary,
        "cv_summary": cv_summary,
    }


if __name__ == "__main__":
    run_experiment()
