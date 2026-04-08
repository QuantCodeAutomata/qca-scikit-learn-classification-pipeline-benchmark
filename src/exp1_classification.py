"""Experiment 1 — Scikit-Learn Classification Pipeline Benchmark.

Evaluates KNeighborsClassifier, SVC, LogisticRegression, and
RandomForestClassifier on the breast-cancer dataset with default and
GridSearchCV-tuned hyperparameters.

# Using scikit-learn classifiers, GridSearchCV, StandardScaler — Context7 confirmed
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.utils import (
    append_results_md,
    plot_bar_comparison,
    plot_confusion_matrices,
)

# ---------------------------------------------------------------------------
# Parameter grids as specified in the methodology
# ---------------------------------------------------------------------------
PARAM_GRIDS: Dict[str, dict] = {
    "KNN": {"n_neighbors": [3, 5, 7, 9]},
    "SVC": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    "LR": {"C": [0.01, 0.1, 1, 10]},
    "RF": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]},
}


def load_and_split_data(
    test_size: float = 0.3, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load breast-cancer dataset and split into train/test sets.

    Parameters
    ----------
    test_size:
        Fraction of data reserved for testing (default 0.3).
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    Tuple of (X_train, X_test, y_train, y_test).
    """
    data = load_breast_cancer()
    X, y = data.data, data.target
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def preprocess(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit StandardScaler on training data and transform both splits.

    Parameters
    ----------
    X_train:
        Raw training features.
    X_test:
        Raw test features.

    Returns
    -------
    Tuple of (X_train_scaled, X_test_scaled).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def build_classifiers() -> Dict[str, object]:
    """Instantiate the four classifiers with default parameters.

    Returns
    -------
    Dict mapping short name → classifier instance.
    """
    return {
        "KNN": KNeighborsClassifier(),
        "SVC": SVC(random_state=42),
        "LR": LogisticRegression(random_state=42, max_iter=10_000),
        "RF": RandomForestClassifier(random_state=42),
    }


def evaluate_classifier(
    clf: object,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    target_names: List[str],
) -> Dict[str, object]:
    """Train a classifier and compute classification metrics on the test set.

    Parameters
    ----------
    clf:
        Scikit-learn classifier instance.
    X_train, X_test:
        Feature matrices.
    y_train, y_test:
        Target arrays.
    target_names:
        Human-readable class labels.

    Returns
    -------
    Dict with keys: accuracy, report (str), confusion_matrix (ndarray).
    """
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, target_names=target_names),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def tune_classifier(
    clf: object,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
) -> GridSearchCV:
    """Run GridSearchCV to find the best hyperparameters.

    Parameters
    ----------
    clf:
        Base classifier.
    param_grid:
        Hyperparameter search space.
    X_train:
        Training features.
    y_train:
        Training labels.
    cv:
        Number of cross-validation folds.

    Returns
    -------
    Fitted GridSearchCV object.
    """
    gs = GridSearchCV(clf, param_grid, cv=cv, n_jobs=-1, scoring="accuracy")
    gs.fit(X_train, y_train)
    return gs


def run_experiment_1() -> pd.DataFrame:
    """Execute the full classification benchmark and return a comparison DataFrame.

    Steps
    -----
    1. Load breast-cancer dataset and split 70/30.
    2. Apply StandardScaler.
    3. Train four classifiers with default parameters; record metrics.
    4. Tune each classifier with GridSearchCV (5-fold CV).
    5. Re-evaluate tuned classifiers; record metrics.
    6. Save confusion matrix heatmaps and bar chart.
    7. Append results to RESULTS.md.

    Returns
    -------
    pd.DataFrame
        Comparison table with default and tuned accuracy for each model.
    """
    # Step 1 — load & split
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Step 2 — preprocess
    X_train_sc, X_test_sc = preprocess(X_train, X_test)

    data = load_breast_cancer()
    target_names = list(data.target_names)

    classifiers = build_classifiers()

    rows: List[dict] = []
    default_cms: Dict[str, np.ndarray] = {}
    tuned_cms: Dict[str, np.ndarray] = {}

    for name, clf in classifiers.items():
        # Step 3 — default evaluation
        default_metrics = evaluate_classifier(
            clf, X_train_sc, X_test_sc, y_train, y_test, target_names
        )
        default_cms[name] = default_metrics["confusion_matrix"]

        # Step 4 & 5 — tune and re-evaluate
        gs = tune_classifier(
            build_classifiers()[name],  # fresh instance for tuning
            PARAM_GRIDS[name],
            X_train_sc,
            y_train,
        )
        tuned_metrics = evaluate_classifier(
            gs.best_estimator_,
            X_train_sc,
            X_test_sc,
            y_train,
            y_test,
            target_names,
        )
        tuned_cms[name] = tuned_metrics["confusion_matrix"]

        # Cross-validation score on tuned best estimator
        cv_scores = cross_val_score(
            gs.best_estimator_, X_train_sc, y_train, cv=5, n_jobs=-1
        )

        rows.append(
            {
                "Model": name,
                "Default Accuracy": round(default_metrics["accuracy"], 4),
                "Tuned Accuracy": round(tuned_metrics["accuracy"], 4),
                "CV Mean (tuned)": round(cv_scores.mean(), 4),
                "CV Std (tuned)": round(cv_scores.std(), 4),
                "Best Params": str(gs.best_params_),
            }
        )

        print(f"\n{'='*60}")
        print(f"[EXP1] {name} — Default Accuracy: {default_metrics['accuracy']:.4f}")
        print(default_metrics["report"])
        print(f"[EXP1] {name} — Tuned Accuracy:   {tuned_metrics['accuracy']:.4f}")
        print(f"[EXP1] {name} — Best Params: {gs.best_params_}")

    results_df = pd.DataFrame(rows)

    # Step 6 — visualisations
    plot_confusion_matrices(
        default_cms,
        target_names,
        title_prefix="Default — ",
        filename="exp1_confusion_matrices_default.png",
    )
    plot_confusion_matrices(
        tuned_cms,
        target_names,
        title_prefix="Tuned — ",
        filename="exp1_confusion_matrices_tuned.png",
    )
    plot_bar_comparison(
        results_df,
        metric_cols=["Default Accuracy", "Tuned Accuracy", "CV Mean (tuned)"],
        group_col="Model",
        title="Exp 1 — Classification Accuracy: Default vs Tuned",
        ylabel="Accuracy",
        filename="exp1_accuracy_comparison.png",
    )

    # Step 7 — persist results
    md_table = results_df.to_markdown(index=False)
    append_results_md(
        "Experiment 1 — Classification Pipeline Benchmark",
        f"### Breast Cancer Dataset — Default vs Tuned Accuracy\n\n{md_table}\n",
    )

    print("\n[EXP1] Results summary:")
    print(results_df.to_string(index=False))
    return results_df
