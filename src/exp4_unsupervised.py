"""
Experiment 4: Scikit-Learn Unsupervised Learning and Dimensionality Reduction
==============================================================================
Applies PCA and KMeans clustering to the Wine dataset, using the elbow method
and silhouette scores to identify the optimal number of clusters.

Methodology (DataCamp Scikit-Learn Cheat Sheet):
  1. Load Wine dataset, standardize with StandardScaler
  2. PCA with n_components=2 and 3 → explained_variance_ratio_
  3. Plot 2D PCA projection colored by true class labels
  4. KMeans for k=2..10 (random_state=42, n_init=10) → record inertia
  5. Plot elbow curve (k vs inertia)
  6. Compute silhouette_score for each k → plot k vs silhouette
  7. Plot cluster assignments in PCA 2D space for optimal k
  8. Crosstab contingency table: discovered clusters vs true labels

# Using scikit-learn PCA, KMeans, silhouette_score — Context7 confirmed
# Using matplotlib/seaborn for visualizations — Context7 confirmed
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

def load_wine_dataframe() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Load the Wine dataset and return features, target, and class names.

    Returns
    -------
    X : pd.DataFrame  (178 samples × 13 features)
    y : pd.Series     Integer class labels (0, 1, 2)
    class_names : list of str
    """
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name="class")
    print(f"Wine dataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(np.unique(y))} classes")
    return X, y, list(wine.target_names)


def standardize_features(X: pd.DataFrame) -> np.ndarray:
    """Apply StandardScaler to all features.

    Parameters
    ----------
    X : pd.DataFrame

    Returns
    -------
    X_scaled : np.ndarray
    """
    # Using scikit-learn StandardScaler — Context7 confirmed
    scaler = StandardScaler()
    return scaler.fit_transform(X)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def apply_pca(
    X_scaled: np.ndarray,
    n_components: int,
) -> Tuple[np.ndarray, PCA]:
    """Fit PCA and return transformed data and fitted PCA object.

    Parameters
    ----------
    X_scaled : np.ndarray
    n_components : int

    Returns
    -------
    X_pca : np.ndarray  shape (n_samples, n_components)
    pca : PCA (fitted)
    """
    # Using scikit-learn PCA — Context7 confirmed
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    print(f"\nPCA (n_components={n_components}):")
    print(f"  Explained variance ratio: {np.round(pca.explained_variance_ratio_, 4)}")
    print(f"  Cumulative explained variance: {np.round(cumulative_var, 4)}")
    return X_pca, pca


# ---------------------------------------------------------------------------
# KMeans clustering
# ---------------------------------------------------------------------------

def run_kmeans_sweep(
    X_scaled: np.ndarray,
    k_range: range = None,
) -> Dict[int, Dict]:
    """Run KMeans for each k in k_range and record inertia and silhouette score.

    Parameters
    ----------
    X_scaled : np.ndarray
    k_range : range
        Default: range(2, 11).

    Returns
    -------
    kmeans_results : dict
        k → {'inertia': float, 'silhouette': float, 'labels': np.ndarray}
    """
    if k_range is None:
        k_range = range(2, 11)

    kmeans_results: Dict[int, Dict] = {}
    print("\n=== KMeans Sweep ===")
    for k in k_range:
        # Using scikit-learn KMeans — Context7 confirmed
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia = km.inertia_
        sil = silhouette_score(X_scaled, labels)
        kmeans_results[k] = {"inertia": inertia, "silhouette": sil, "labels": labels}
        print(f"  k={k:2d}: inertia={inertia:.2f}, silhouette={sil:.4f}")
    return kmeans_results


def find_optimal_k(kmeans_results: Dict[int, Dict]) -> int:
    """Identify optimal k as the one with the highest silhouette score.

    Parameters
    ----------
    kmeans_results : dict

    Returns
    -------
    int  Optimal k value.
    """
    optimal_k = max(kmeans_results, key=lambda k: kmeans_results[k]["silhouette"])
    print(f"\nOptimal k (max silhouette): {optimal_k} "
          f"(silhouette={kmeans_results[optimal_k]['silhouette']:.4f})")
    return optimal_k


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_pca_true_labels(
    X_pca_2d: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    output_dir: str = "results",
) -> None:
    """Scatter plot of 2D PCA projection colored by true class labels.

    Parameters
    ----------
    X_pca_2d : np.ndarray  shape (n_samples, 2)
    y : np.ndarray  True class labels.
    class_names : list of str
    output_dir : str
    """
    os.makedirs(output_dir, exist_ok=True)
    palette = sns.color_palette("Set1", n_colors=len(class_names))
    fig, ax = plt.subplots(figsize=(7, 5))
    for cls_idx, cls_name in enumerate(class_names):
        mask = y == cls_idx
        ax.scatter(
            X_pca_2d[mask, 0], X_pca_2d[mask, 1],
            label=cls_name, color=palette[cls_idx], alpha=0.8, s=60, edgecolors="k", linewidths=0.4,
        )
    ax.set_title("Exp 4 – PCA 2D Projection (True Class Labels)", fontsize=13)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="Class")
    plt.tight_layout()
    path = os.path.join(output_dir, "exp4_pca_true_labels.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved PCA true labels plot → {path}")


def plot_elbow_curve(
    kmeans_results: Dict[int, Dict],
    output_dir: str = "results",
) -> None:
    """Plot the elbow curve (k vs inertia).

    Parameters
    ----------
    kmeans_results : dict
    output_dir : str
    """
    os.makedirs(output_dir, exist_ok=True)
    k_values = sorted(kmeans_results.keys())
    inertias = [kmeans_results[k]["inertia"] for k in k_values]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_values, inertias, "o-", color="#4C72B0", linewidth=2, markersize=7)
    ax.set_title("Exp 4 – Elbow Curve (KMeans Inertia vs k)", fontsize=13)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    ax.set_xticks(k_values)
    ax.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(output_dir, "exp4_elbow_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved elbow curve plot → {path}")


def plot_silhouette_scores(
    kmeans_results: Dict[int, Dict],
    output_dir: str = "results",
) -> None:
    """Plot silhouette scores for each k.

    Parameters
    ----------
    kmeans_results : dict
    output_dir : str
    """
    os.makedirs(output_dir, exist_ok=True)
    k_values = sorted(kmeans_results.keys())
    silhouettes = [kmeans_results[k]["silhouette"] for k in k_values]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_values, silhouettes, "s-", color="#DD8452", linewidth=2, markersize=7)
    ax.set_title("Exp 4 – Silhouette Score vs k", fontsize=13)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_xticks(k_values)
    ax.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(output_dir, "exp4_silhouette_scores.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved silhouette scores plot → {path}")


def plot_cluster_assignments(
    X_pca_2d: np.ndarray,
    cluster_labels: np.ndarray,
    optimal_k: int,
    output_dir: str = "results",
) -> None:
    """Scatter plot of PCA 2D projection colored by KMeans cluster assignments.

    Parameters
    ----------
    X_pca_2d : np.ndarray  shape (n_samples, 2)
    cluster_labels : np.ndarray  Cluster assignments for optimal k.
    optimal_k : int
    output_dir : str
    """
    os.makedirs(output_dir, exist_ok=True)
    palette = sns.color_palette("Set2", n_colors=optimal_k)
    fig, ax = plt.subplots(figsize=(7, 5))
    for cluster_id in range(optimal_k):
        mask = cluster_labels == cluster_id
        ax.scatter(
            X_pca_2d[mask, 0], X_pca_2d[mask, 1],
            label=f"Cluster {cluster_id}",
            color=palette[cluster_id], alpha=0.8, s=60, edgecolors="k", linewidths=0.4,
        )
    ax.set_title(f"Exp 4 – KMeans Cluster Assignments (k={optimal_k})", fontsize=13)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="Cluster")
    plt.tight_layout()
    path = os.path.join(output_dir, "exp4_cluster_assignments.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved cluster assignments plot → {path}")


# ---------------------------------------------------------------------------
# Contingency table
# ---------------------------------------------------------------------------

def build_contingency_table(
    y_true: np.ndarray,
    cluster_labels: np.ndarray,
    class_names: List[str],
) -> pd.DataFrame:
    """Build a crosstab contingency table: true labels vs cluster assignments.

    Parameters
    ----------
    y_true : np.ndarray  True class labels.
    cluster_labels : np.ndarray  KMeans cluster assignments.
    class_names : list of str

    Returns
    -------
    pd.DataFrame  Contingency table.
    """
    df = pd.DataFrame({"True Class": y_true, "Cluster": cluster_labels})
    df["True Class"] = df["True Class"].map(
        {i: name for i, name in enumerate(class_names)}
    )
    contingency = pd.crosstab(df["True Class"], df["Cluster"], margins=True)
    print("\n=== Contingency Table (True Labels vs Clusters) ===")
    print(contingency.to_string())
    return contingency


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_pca_summary(pca_2d: PCA, pca_3d: PCA) -> pd.DataFrame:
    """Summarize PCA explained variance for 2 and 3 components.

    Parameters
    ----------
    pca_2d : PCA (fitted, n_components=2)
    pca_3d : PCA (fitted, n_components=3)

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for n_comp, pca in [(2, pca_2d), (3, pca_3d)]:
        rows.append(
            {
                "n_components": n_comp,
                "Explained Variance Ratio": str(np.round(pca.explained_variance_ratio_, 4)),
                "Cumulative Variance": round(float(np.sum(pca.explained_variance_ratio_)), 4),
            }
        )
    df = pd.DataFrame(rows).set_index("n_components")
    print("\n=== PCA Summary ===")
    print(df.to_string())
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_experiment(output_dir: str = "results") -> Dict:
    """Execute the full Exp 4 unsupervised learning study.

    Parameters
    ----------
    output_dir : str
        Directory for saving plots and artefacts.

    Returns
    -------
    dict
        Contains PCA objects, KMeans results, optimal k, and contingency table.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1 – Load data and standardize
    X, y, class_names = load_wine_dataframe()
    X_scaled = standardize_features(X)

    # Step 2 – PCA with 2 and 3 components
    X_pca_2d, pca_2d = apply_pca(X_scaled, n_components=2)
    X_pca_3d, pca_3d = apply_pca(X_scaled, n_components=3)

    # Step 3 – Plot 2D PCA with true labels
    plot_pca_true_labels(X_pca_2d, y.values, class_names, output_dir)

    # Step 4 – KMeans sweep k=2..10
    kmeans_results = run_kmeans_sweep(X_scaled, k_range=range(2, 11))

    # Step 5 – Elbow curve
    plot_elbow_curve(kmeans_results, output_dir)

    # Step 6 – Silhouette scores
    plot_silhouette_scores(kmeans_results, output_dir)

    # Step 7 – Optimal k and cluster assignment plot
    optimal_k = find_optimal_k(kmeans_results)
    optimal_labels = kmeans_results[optimal_k]["labels"]
    plot_cluster_assignments(X_pca_2d, optimal_labels, optimal_k, output_dir)

    # Step 8 – Contingency table
    contingency = build_contingency_table(y.values, optimal_labels, class_names)
    contingency.to_csv(os.path.join(output_dir, "exp4_contingency_table.csv"))

    # PCA summary
    pca_summary = build_pca_summary(pca_2d, pca_3d)
    pca_summary.to_csv(os.path.join(output_dir, "exp4_pca_summary.csv"))

    # KMeans summary
    kmeans_summary_rows = [
        {
            "k": k,
            "Inertia": round(v["inertia"], 2),
            "Silhouette Score": round(v["silhouette"], 4),
        }
        for k, v in sorted(kmeans_results.items())
    ]
    kmeans_summary = pd.DataFrame(kmeans_summary_rows).set_index("k")
    kmeans_summary.to_csv(os.path.join(output_dir, "exp4_kmeans_summary.csv"))

    return {
        "pca_2d": pca_2d,
        "pca_3d": pca_3d,
        "kmeans_results": kmeans_results,
        "optimal_k": optimal_k,
        "contingency_table": contingency,
        "pca_summary": pca_summary,
        "kmeans_summary": kmeans_summary,
    }


if __name__ == "__main__":
    run_experiment()
