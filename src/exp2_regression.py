"""
Experiment 2: Scikit-Learn Regression Pipeline Benchmark
=========================================================
Implements a complete supervised regression workflow on the California Housing
dataset, comparing LinearRegression, Ridge, and RandomForestRegressor.

Methodology (DataCamp Scikit-Learn Cheat Sheet):
  1. Load fetch_california_housing() → pandas DataFrame
  2. EDA: describe statistics, missing values, target distribution
  3. Preprocess: StandardScaler on all features
  4. Train: LinearRegression, Ridge(alpha=1.0), RandomForestRegressor(n_estimators=100, random_state=42)
  5. Predict on test set
  6. Evaluate: MAE, MSE, RMSE, R²
  7. Plot predicted vs actual scatter plots
  8. Summarize in comparison table

# Using scikit-learn regressors and metrics — Context7 confirmed
# Using matplotlib/seaborn for scatter plots — Context7 confirmed
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Data loading & EDA
# ---------------------------------------------------------------------------

def load_california_housing_dataframe() -> Tuple[pd.DataFrame, pd.Series]:
    """Load the California Housing dataset and return features and target.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Continuous target (median house value in $100k units).
    """
    # Using sklearn.datasets.fetch_california_housing — Context7 confirmed
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name="MedHouseVal")
    return X, y


def exploratory_data_analysis(X: pd.DataFrame, y: pd.Series, output_dir: str = "results") -> Dict:
    """Perform EDA: descriptive statistics, missing values, target distribution.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    output_dir : str
        Directory to save the target distribution plot.

    Returns
    -------
    dict
        EDA summary dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)
    eda = {
        "shape": X.shape,
        "missing_values": X.isnull().sum().to_dict(),
        "describe": X.describe(),
        "target_describe": y.describe(),
    }
    print("=== EDA: California Housing Dataset ===")
    print(f"Shape: {eda['shape']}")
    print(f"Missing values: {eda['missing_values']}")
    print(f"\nDescriptive statistics:\n{X.describe()}")
    print(f"\nTarget statistics:\n{y.describe()}\n")

    # Plot target distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_title("California Housing – Target Distribution (Median House Value)")
    ax.set_xlabel("Median House Value ($100k)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    path = os.path.join(output_dir, "exp2_target_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved target distribution plot → {path}")
    return eda


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply StandardScaler: fit on training data, transform both splits.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    X_test : pd.DataFrame
        Test feature matrix.

    Returns
    -------
    X_train_scaled : np.ndarray
    X_test_scaled : np.ndarray
    """
    # Using scikit-learn StandardScaler — Context7 confirmed
    scaler = StandardScaler()
    X_train_scaled = scaler.fit(X_train).transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


# ---------------------------------------------------------------------------
# Model training & evaluation
# ---------------------------------------------------------------------------

def build_regressors() -> Dict[str, object]:
    """Instantiate the three regressors with specified hyperparameters.

    Returns
    -------
    dict
        Mapping of regressor name → estimator instance.
    """
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
    }


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute MAE, MSE, RMSE, and R² for a set of predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth target values.
    y_pred : np.ndarray
        Model predictions.

    Returns
    -------
    dict
        Keys: mae, mse, rmse, r2.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))   # RMSE = sqrt(MSE)
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


def train_and_evaluate(
    regressors: Dict[str, object],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Dict]:
    """Train each regressor and compute evaluation metrics.

    Parameters
    ----------
    regressors : dict
        Name → estimator mapping.
    X_train, X_test : np.ndarray
        Scaled feature arrays.
    y_train, y_test : np.ndarray
        Target arrays.

    Returns
    -------
    results : dict
        Per-regressor dict with keys: metrics, y_pred.
    """
    results: Dict[str, Dict] = {}
    for name, reg in regressors.items():
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        metrics = compute_regression_metrics(y_test, y_pred)
        results[name] = {"metrics": metrics, "y_pred": y_pred}
        print(f"\n--- {name} ---")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  MSE:  {metrics['mse']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  R²:   {metrics['r2']:.4f}")
    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_predicted_vs_actual(
    results: Dict[str, Dict],
    y_test: np.ndarray,
    output_dir: str = "results",
) -> None:
    """Plot predicted vs actual scatter plots for each regressor.

    Parameters
    ----------
    results : dict
        Output from train_and_evaluate().
    y_test : np.ndarray
        True target values.
    output_dir : str
        Directory to save the figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        y_pred = res["y_pred"]
        metrics = res["metrics"]
        ax.scatter(y_test, y_pred, alpha=0.3, s=10, color="steelblue")
        # Identity line (perfect predictions)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
        ax.set_title(
            f"{name}\nRMSE={metrics['rmse']:.3f}  R²={metrics['r2']:.3f}",
            fontsize=11,
        )
        ax.set_xlabel("Actual Value")
        ax.set_ylabel("Predicted Value")
        ax.legend(fontsize=9)

    plt.suptitle("Exp 2 – Predicted vs Actual (California Housing)", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "exp2_predicted_vs_actual.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved predicted vs actual plot → {path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """Build a comparison DataFrame from per-regressor metrics.

    Parameters
    ----------
    results : dict
        Output from train_and_evaluate().

    Returns
    -------
    pd.DataFrame
        Rows = regressors, columns = MAE / MSE / RMSE / R².
    """
    rows = []
    for name, res in results.items():
        m = res["metrics"]
        rows.append(
            {
                "Regressor": name,
                "MAE": round(m["mae"], 4),
                "MSE": round(m["mse"], 4),
                "RMSE": round(m["rmse"], 4),
                "R²": round(m["r2"], 4),
            }
        )
    df = pd.DataFrame(rows).set_index("Regressor")
    print("\n=== Comparison Table ===")
    print(df.to_string())
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_experiment(output_dir: str = "results") -> Dict:
    """Execute the full Exp 2 regression pipeline.

    Parameters
    ----------
    output_dir : str
        Directory for saving plots and artefacts.

    Returns
    -------
    dict
        Contains 'results' (per-model metrics) and 'comparison_table'.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1 – Load data
    X, y = load_california_housing_dataframe()

    # Step 2 – EDA
    exploratory_data_analysis(X, y, output_dir)

    # Step 3 – Train/test split (70/30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    # Step 4 – Preprocessing
    X_train_scaled, X_test_scaled = preprocess_features(X_train, X_test)

    # Step 5 – Train & evaluate
    regressors = build_regressors()
    results = train_and_evaluate(
        regressors, X_train_scaled, X_test_scaled,
        y_train.values, y_test.values
    )

    # Step 6 – Visualize predicted vs actual
    plot_predicted_vs_actual(results, y_test.values, output_dir)

    # Step 7 – Comparison table
    comparison_table = build_comparison_table(results)
    comparison_table.to_csv(os.path.join(output_dir, "exp2_comparison_table.csv"))

    return {"results": results, "comparison_table": comparison_table}


if __name__ == "__main__":
    run_experiment()
