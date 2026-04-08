"""
Experiment 1: Scikit-Learn Classification Pipeline Benchmark
=============================================================
Implements a complete supervised classification workflow on the Iris dataset,
comparing LogisticRegression, KNeighborsClassifier, and SVC.

Methodology (DataCamp Scikit-Learn Cheat Sheet):
  1. Load Iris dataset → pandas DataFrame
  2. EDA: shapes, dtypes, missing values, class distributions
  3. Preprocess: StandardScaler (fit on train, transform on test)
  4. Train: LogisticRegression, KNeighborsClassifier, SVC (default params)
  5. Predict on test set
  6. Evaluate: accuracy_score, classification_report, confusion_matrix
  7. Visualize confusion matrices as heatmaps
  8. Summarize in comparison table

# Using scikit-learn StandardScaler, classifiers, metrics — Context7 confirmed
# Using seaborn heatmap for confusion matrix visualization — Context7 confirmed
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Data loading & EDA
# ---------------------------------------------------------------------------

def load_iris_dataframe() -> Tuple[pd.DataFrame, pd.Series]:
    """Load the Iris dataset and return features DataFrame and target Series.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with column names from the dataset.
    y : pd.Series
        Integer-encoded target labels.
    """
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")
    return X, y


def exploratory_data_analysis(X: pd.DataFrame, y: pd.Series) -> Dict:
    """Perform basic EDA on the Iris dataset.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels.

    Returns
    -------
    dict
        Dictionary containing shape, dtypes, missing values, and class counts.
    """
    eda = {
        "shape": X.shape,
        "dtypes": X.dtypes.to_dict(),
        "missing_values": X.isnull().sum().to_dict(),
        "class_distribution": y.value_counts().to_dict(),
        "describe": X.describe(),
    }
    print("=== EDA: Iris Dataset ===")
    print(f"Shape: {eda['shape']}")
    print(f"Dtypes:\n{X.dtypes}")
    print(f"Missing values: {eda['missing_values']}")
    print(f"Class distribution:\n{y.value_counts()}")
    print(f"\nDescriptive statistics:\n{X.describe()}\n")
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
        Scaled training features.
    X_test_scaled : np.ndarray
        Scaled test features.
    """
    # Using scikit-learn StandardScaler — Context7 confirmed
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


# ---------------------------------------------------------------------------
# Model training & evaluation
# ---------------------------------------------------------------------------

def build_classifiers() -> Dict[str, object]:
    """Instantiate the three classifiers with default hyperparameters.

    Returns
    -------
    dict
        Mapping of classifier name → estimator instance.
    """
    return {
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=200),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "SVC": SVC(random_state=42),
    }


def train_and_evaluate(
    classifiers: Dict[str, object],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Dict]:
    """Train each classifier and compute evaluation metrics.

    Parameters
    ----------
    classifiers : dict
        Name → estimator mapping.
    X_train, X_test : np.ndarray
        Scaled feature arrays.
    y_train, y_test : np.ndarray
        Target label arrays.

    Returns
    -------
    results : dict
        Per-classifier dict with keys: accuracy, report, confusion_matrix, y_pred.
    """
    results: Dict[str, Dict] = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {
            "accuracy": acc,
            "report": report,
            "confusion_matrix": cm,
            "y_pred": y_pred,
        }
        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_confusion_matrices(
    results: Dict[str, Dict],
    class_names: List[str],
    output_dir: str = "results",
) -> None:
    """Plot confusion matrices as annotated heatmaps for each classifier.

    Parameters
    ----------
    results : dict
        Output from train_and_evaluate().
    class_names : list of str
        Human-readable class labels.
    output_dir : str
        Directory to save the figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = res["confusion_matrix"]
        # Using seaborn heatmap — Context7 confirmed
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_title(f"{name}\nAccuracy: {res['accuracy']:.4f}", fontsize=12)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.suptitle("Exp 1 – Confusion Matrices (Iris Dataset)", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "exp1_confusion_matrices.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot → {path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """Build a comparison DataFrame from per-classifier metrics.

    Parameters
    ----------
    results : dict
        Output from train_and_evaluate().

    Returns
    -------
    pd.DataFrame
        Rows = classifiers, columns = accuracy / precision / recall / f1.
    """
    rows = []
    for name, res in results.items():
        report = res["report"]
        rows.append(
            {
                "Classifier": name,
                "Accuracy": round(res["accuracy"], 4),
                "Precision (macro)": round(report["macro avg"]["precision"], 4),
                "Recall (macro)": round(report["macro avg"]["recall"], 4),
                "F1-Score (macro)": round(report["macro avg"]["f1-score"], 4),
            }
        )
    df = pd.DataFrame(rows).set_index("Classifier")
    print("\n=== Comparison Table ===")
    print(df.to_string())
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_experiment(output_dir: str = "results") -> Dict:
    """Execute the full Exp 1 classification pipeline.

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
    X, y = load_iris_dataframe()
    iris = load_iris()
    class_names = list(iris.target_names)

    # Step 2 – EDA
    exploratory_data_analysis(X, y)

    # Step 3 – Train/test split (70/30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Step 4 – Preprocessing
    X_train_scaled, X_test_scaled = preprocess_features(X_train, X_test)

    # Step 5 – Train & evaluate
    classifiers = build_classifiers()
    results = train_and_evaluate(
        classifiers, X_train_scaled, X_test_scaled,
        y_train.values, y_test.values
    )

    # Step 6 – Visualize confusion matrices
    plot_confusion_matrices(results, class_names, output_dir)

    # Step 7 – Comparison table
    comparison_table = build_comparison_table(results)
    comparison_table.to_csv(os.path.join(output_dir, "exp1_comparison_table.csv"))

    return {"results": results, "comparison_table": comparison_table}


if __name__ == "__main__":
    run_experiment()
