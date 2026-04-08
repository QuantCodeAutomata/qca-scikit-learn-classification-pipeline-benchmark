"""Experiment 2 — Scikit-Learn Regression Pipeline Benchmark.

Evaluates LinearRegression, Ridge, Lasso, and RandomForestRegressor on the
diabetes dataset with default and GridSearchCV-tuned hyperparameters.

# Using scikit-learn regressors, GridSearchCV, StandardScaler — Context7 confirmed
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import (
    append_results_md,
    plot_bar_comparison,
    plot_scatter_actual_vs_predicted,
)

# ---------------------------------------------------------------------------
# Parameter grids as specified in the methodology
# ---------------------------------------------------------------------------
PARAM_GRIDS: Dict[str, dict] = {
    "Ridge": {"alpha": [0.01, 0.1, 1, 10, 100]},
    "Lasso": {"alpha": [0.001, 0.01, 0.1, 1]},
    "RF": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]},
}


def load_and_split_data(
    test_size: float = 0.3, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load diabetes dataset and split into train/test sets.

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
    data = load_diabetes()
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


def compute_regression_metrics(
    y_test: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute MSE, RMSE, MAE, and R² for a set of predictions.

    Parameters
    ----------
    y_test:
        Ground-truth target values.
    y_pred:
        Model predictions.

    Returns
    -------
    Dict with keys: MSE, RMSE, MAE, R2.
    """
    mse = mean_squared_error(y_test, y_pred)
    return {
        "MSE": round(mse, 4),
        "RMSE": round(np.sqrt(mse), 4),
        "MAE": round(mean_absolute_error(y_test, y_pred), 4),
        "R2": round(r2_score(y_test, y_pred), 4),
    }


def build_regressors() -> Dict[str, object]:
    """Instantiate the four regressors with default parameters.

    Returns
    -------
    Dict mapping short name → regressor instance.
    """
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "RF": RandomForestRegressor(random_state=42),
    }


def tune_regressor(
    reg: object,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
) -> GridSearchCV:
    """Run GridSearchCV to find the best hyperparameters for a regressor.

    Parameters
    ----------
    reg:
        Base regressor.
    param_grid:
        Hyperparameter search space.
    X_train:
        Training features.
    y_train:
        Training targets.
    cv:
        Number of cross-validation folds.

    Returns
    -------
    Fitted GridSearchCV object.
    """
    gs = GridSearchCV(
        reg,
        param_grid,
        cv=cv,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
    )
    gs.fit(X_train, y_train)
    return gs


def run_experiment_2() -> pd.DataFrame:
    """Execute the full regression benchmark and return a comparison DataFrame.

    Steps
    -----
    1. Load diabetes dataset and split 70/30.
    2. Apply StandardScaler.
    3. Train four regressors with default parameters; record metrics.
    4. Tune Ridge, Lasso, RF with GridSearchCV (5-fold CV).
    5. Re-evaluate tuned regressors; record metrics.
    6. Save scatter plots and bar chart.
    7. Append results to RESULTS.md.

    Returns
    -------
    pd.DataFrame
        Comparison table with default and tuned metrics for each model.
    """
    # Step 1 — load & split
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Step 2 — preprocess
    X_train_sc, X_test_sc = preprocess(X_train, X_test)

    regressors = build_regressors()

    rows: List[dict] = []
    default_preds: Dict[str, np.ndarray] = {}
    tuned_preds: Dict[str, np.ndarray] = {}

    for name, reg in regressors.items():
        # Step 3 — default evaluation
        reg.fit(X_train_sc, y_train)
        y_pred_default = reg.predict(X_test_sc)
        default_metrics = compute_regression_metrics(y_test, y_pred_default)
        default_preds[name] = y_pred_default

        # Step 4 & 5 — tune (only Ridge, Lasso, RF) and re-evaluate
        if name in PARAM_GRIDS:
            gs = tune_regressor(
                build_regressors()[name],
                PARAM_GRIDS[name],
                X_train_sc,
                y_train,
            )
            y_pred_tuned = gs.best_estimator_.predict(X_test_sc)
            tuned_metrics = compute_regression_metrics(y_test, y_pred_tuned)
            tuned_preds[f"{name} (tuned)"] = y_pred_tuned
            best_params = str(gs.best_params_)
        else:
            # LinearRegression has no tunable hyperparameters in this study
            tuned_metrics = default_metrics
            tuned_preds[f"{name} (tuned)"] = y_pred_default
            best_params = "N/A"

        rows.append(
            {
                "Model": name,
                "Default RMSE": default_metrics["RMSE"],
                "Tuned RMSE": tuned_metrics["RMSE"],
                "Default R2": default_metrics["R2"],
                "Tuned R2": tuned_metrics["R2"],
                "Default MAE": default_metrics["MAE"],
                "Tuned MAE": tuned_metrics["MAE"],
                "Best Params": best_params,
            }
        )

        print(f"\n{'='*60}")
        print(f"[EXP2] {name}")
        print(f"  Default → RMSE={default_metrics['RMSE']}, R²={default_metrics['R2']}")
        if name in PARAM_GRIDS:
            print(f"  Tuned   → RMSE={tuned_metrics['RMSE']}, R²={tuned_metrics['R2']}")
            print(f"  Best Params: {best_params}")

    results_df = pd.DataFrame(rows)

    # Step 6 — visualisations
    # Scatter: default predictions
    plot_scatter_actual_vs_predicted(
        y_test,
        default_preds,
        title="Exp 2 — Default Regressors: Actual vs Predicted",
        filename="exp2_scatter_default.png",
    )
    # Scatter: tuned predictions
    plot_scatter_actual_vs_predicted(
        y_test,
        tuned_preds,
        title="Exp 2 — Tuned Regressors: Actual vs Predicted",
        filename="exp2_scatter_tuned.png",
    )
    # Bar chart: RMSE comparison
    plot_bar_comparison(
        results_df,
        metric_cols=["Default RMSE", "Tuned RMSE"],
        group_col="Model",
        title="Exp 2 — RMSE: Default vs Tuned",
        ylabel="RMSE",
        filename="exp2_rmse_comparison.png",
    )
    # Bar chart: R² comparison
    plot_bar_comparison(
        results_df,
        metric_cols=["Default R2", "Tuned R2"],
        group_col="Model",
        title="Exp 2 — R²: Default vs Tuned",
        ylabel="R²",
        filename="exp2_r2_comparison.png",
    )

    # Step 7 — persist results
    md_table = results_df.to_markdown(index=False)
    append_results_md(
        "Experiment 2 — Regression Pipeline Benchmark",
        f"### Diabetes Dataset — Default vs Tuned Metrics\n\n{md_table}\n",
    )

    print("\n[EXP2] Results summary:")
    print(results_df.to_string(index=False))
    return results_df
