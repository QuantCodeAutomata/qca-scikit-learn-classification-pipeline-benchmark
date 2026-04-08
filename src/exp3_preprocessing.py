"""Experiment 3 — Scikit-Learn Preprocessing and Feature Engineering Impact Study.

Isolates the effect of different preprocessing strategies on SVC performance
using the wine dataset (multi-class classification).

Sub-experiments:
  A — Scaling: no scaling vs StandardScaler vs MinMaxScaler
  B — Train/Test split ratio: 60/40, 70/30, 80/20
  C — Categorical encoding: LabelEncoder vs one-hot (pd.get_dummies)

# Using scikit-learn preprocessing, Pipeline, cross_val_score — Context7 confirmed
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from src.utils import append_results_md, plot_bar_comparison

# ---------------------------------------------------------------------------
# Control classifier (fixed throughout all sub-experiments)
# ---------------------------------------------------------------------------
CONTROL_CLF_PARAMS = {"kernel": "rbf", "C": 1, "random_state": 42}


def _make_pipeline(scaler: object | None) -> Pipeline:
    """Build a Pipeline with an optional scaler and the control SVC.

    Parameters
    ----------
    scaler:
        A fitted-or-unfitted scaler instance, or None for no scaling.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    steps = []
    if scaler is not None:
        steps.append(("scaler", scaler))
    steps.append(("svc", SVC(**CONTROL_CLF_PARAMS)))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Experiment A — Scaling
# ---------------------------------------------------------------------------

def experiment_a_scaling(
    X: np.ndarray, y: np.ndarray, cv: int = 5, random_state: int = 42
) -> pd.DataFrame:
    """Compare SVC performance under three scaling conditions.

    Conditions: no scaling, StandardScaler, MinMaxScaler.
    Uses a fixed 70/30 train/test split and 5-fold cross-validation.

    Parameters
    ----------
    X:
        Feature matrix.
    y:
        Target labels.
    cv:
        Number of cross-validation folds.
    random_state:
        Seed for train_test_split.

    Returns
    -------
    pd.DataFrame with columns: Scaling, Test Accuracy, CV Mean, CV Std.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )

    conditions = {
        "No Scaling": None,
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
    }

    rows: List[dict] = []
    for label, scaler in conditions.items():
        pipe = _make_pipeline(scaler)
        pipe.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, pipe.predict(X_test))

        # Cross-validation on full dataset for stability estimate
        cv_scores = cross_val_score(
            _make_pipeline(scaler.__class__() if scaler is not None else None),
            X,
            y,
            cv=cv,
            n_jobs=-1,
        )
        rows.append(
            {
                "Scaling": label,
                "Test Accuracy": round(test_acc, 4),
                "CV Mean": round(cv_scores.mean(), 4),
                "CV Std": round(cv_scores.std(), 4),
            }
        )
        print(
            f"[EXP3-A] {label:20s} → Test={test_acc:.4f}, "
            f"CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}"
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiment B — Train/Test Split Ratio
# ---------------------------------------------------------------------------

def experiment_b_split_ratio(
    X: np.ndarray, y: np.ndarray, cv: int = 5, random_state: int = 42
) -> pd.DataFrame:
    """Evaluate SVC (with StandardScaler) across three train/test split ratios.

    Ratios tested: 60/40, 70/30, 80/20.

    Parameters
    ----------
    X:
        Feature matrix.
    y:
        Target labels.
    cv:
        Number of cross-validation folds.
    random_state:
        Seed for train_test_split.

    Returns
    -------
    pd.DataFrame with columns: Split, Test Accuracy, CV Mean, CV Std.
    """
    split_ratios = [0.4, 0.3, 0.2]  # test_size values → 60/40, 70/30, 80/20
    split_labels = ["60/40", "70/30", "80/20"]

    rows: List[dict] = []
    for test_size, label in zip(split_ratios, split_labels):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        pipe = _make_pipeline(StandardScaler())
        pipe.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, pipe.predict(X_test))

        cv_scores = cross_val_score(
            _make_pipeline(StandardScaler()), X, y, cv=cv, n_jobs=-1
        )
        rows.append(
            {
                "Split": label,
                "Test Accuracy": round(test_acc, 4),
                "CV Mean": round(cv_scores.mean(), 4),
                "CV Std": round(cv_scores.std(), 4),
            }
        )
        print(
            f"[EXP3-B] Split {label:6s} → Test={test_acc:.4f}, "
            f"CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}"
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiment C — Categorical Encoding
# ---------------------------------------------------------------------------

def _build_synthetic_dataset(
    n_samples: int = 500, n_numeric: int = 4, n_categories: int = 3, random_state: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Construct a synthetic dataset with numeric and categorical features.

    Parameters
    ----------
    n_samples:
        Number of samples.
    n_numeric:
        Number of numeric features.
    n_categories:
        Number of levels for the categorical feature.
    random_state:
        Random seed.

    Returns
    -------
    Tuple of (DataFrame with features, target array).
    """
    rng = np.random.default_rng(random_state)
    numeric_data = rng.standard_normal((n_samples, n_numeric))
    cat_feature = rng.integers(0, n_categories, size=n_samples)
    cat_labels = np.array(["cat_A", "cat_B", "cat_C"])[cat_feature]

    df = pd.DataFrame(
        numeric_data, columns=[f"num_{i}" for i in range(n_numeric)]
    )
    df["category"] = cat_labels

    # Binary target: 1 if sum of numeric features > 0, else 0
    y = (numeric_data.sum(axis=1) > 0).astype(int)
    return df, y


def experiment_c_encoding(
    cv: int = 5, random_state: int = 42
) -> pd.DataFrame:
    """Compare LabelEncoder vs one-hot encoding on a synthetic mixed dataset.

    Parameters
    ----------
    cv:
        Number of cross-validation folds.
    random_state:
        Seed for train_test_split.

    Returns
    -------
    pd.DataFrame with columns: Encoding, Test Accuracy, CV Mean, CV Std.
    """
    df, y = _build_synthetic_dataset(random_state=random_state)

    rows: List[dict] = []

    # --- LabelEncoder ---
    df_le = df.copy()
    le = LabelEncoder()
    df_le["category"] = le.fit_transform(df_le["category"])
    X_le = df_le.values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X_le, y, test_size=0.3, random_state=random_state
    )
    pipe_le = _make_pipeline(StandardScaler())
    pipe_le.fit(X_train, y_train)
    acc_le = accuracy_score(y_test, pipe_le.predict(X_test))
    cv_le = cross_val_score(
        _make_pipeline(StandardScaler()), X_le, y, cv=cv, n_jobs=-1
    )
    rows.append(
        {
            "Encoding": "LabelEncoder",
            "Test Accuracy": round(acc_le, 4),
            "CV Mean": round(cv_le.mean(), 4),
            "CV Std": round(cv_le.std(), 4),
        }
    )
    print(
        f"[EXP3-C] LabelEncoder → Test={acc_le:.4f}, "
        f"CV={cv_le.mean():.4f}±{cv_le.std():.4f}"
    )

    # --- One-hot encoding (pd.get_dummies) ---
    df_ohe = pd.get_dummies(df, columns=["category"])
    X_ohe = df_ohe.values.astype(float)

    X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
        X_ohe, y, test_size=0.3, random_state=random_state
    )
    pipe_ohe = _make_pipeline(StandardScaler())
    pipe_ohe.fit(X_train_o, y_train_o)
    acc_ohe = accuracy_score(y_test_o, pipe_ohe.predict(X_test_o))
    cv_ohe = cross_val_score(
        _make_pipeline(StandardScaler()), X_ohe, y, cv=cv, n_jobs=-1
    )
    rows.append(
        {
            "Encoding": "One-Hot (get_dummies)",
            "Test Accuracy": round(acc_ohe, 4),
            "CV Mean": round(cv_ohe.mean(), 4),
            "CV Std": round(cv_ohe.std(), 4),
        }
    )
    print(
        f"[EXP3-C] One-Hot     → Test={acc_ohe:.4f}, "
        f"CV={cv_ohe.mean():.4f}±{cv_ohe.std():.4f}"
    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_experiment_3() -> Dict[str, pd.DataFrame]:
    """Execute all three preprocessing sub-experiments.

    Returns
    -------
    Dict with keys 'scaling', 'split_ratio', 'encoding' mapping to result DataFrames.
    """
    data = load_wine()
    X, y = data.data, data.target

    print("\n" + "=" * 60)
    print("[EXP3] Sub-experiment A — Scaling")
    df_a = experiment_a_scaling(X, y)

    print("\n" + "=" * 60)
    print("[EXP3] Sub-experiment B — Train/Test Split Ratio")
    df_b = experiment_b_split_ratio(X, y)

    print("\n" + "=" * 60)
    print("[EXP3] Sub-experiment C — Categorical Encoding")
    df_c = experiment_c_encoding()

    # Visualisations
    plot_bar_comparison(
        df_a,
        metric_cols=["Test Accuracy", "CV Mean"],
        group_col="Scaling",
        title="Exp 3A — Scaling Strategy Impact on SVC",
        ylabel="Accuracy",
        filename="exp3a_scaling_comparison.png",
    )
    plot_bar_comparison(
        df_b,
        metric_cols=["Test Accuracy", "CV Mean"],
        group_col="Split",
        title="Exp 3B — Train/Test Split Ratio Impact on SVC",
        ylabel="Accuracy",
        filename="exp3b_split_comparison.png",
    )
    plot_bar_comparison(
        df_c,
        metric_cols=["Test Accuracy", "CV Mean"],
        group_col="Encoding",
        title="Exp 3C — Encoding Strategy Impact on SVC",
        ylabel="Accuracy",
        filename="exp3c_encoding_comparison.png",
    )

    # Persist results
    md_a = df_a.to_markdown(index=False)
    md_b = df_b.to_markdown(index=False)
    md_c = df_c.to_markdown(index=False)
    append_results_md(
        "Experiment 3 — Preprocessing & Feature Engineering Impact Study",
        (
            "### 3A — Scaling Strategy (Wine Dataset, SVC control)\n\n"
            f"{md_a}\n\n"
            "### 3B — Train/Test Split Ratio\n\n"
            f"{md_b}\n\n"
            "### 3C — Categorical Encoding (Synthetic Dataset)\n\n"
            f"{md_c}\n"
        ),
    )

    print("\n[EXP3] Sub-experiment A:")
    print(df_a.to_string(index=False))
    print("\n[EXP3] Sub-experiment B:")
    print(df_b.to_string(index=False))
    print("\n[EXP3] Sub-experiment C:")
    print(df_c.to_string(index=False))

    return {"scaling": df_a, "split_ratio": df_b, "encoding": df_c}
