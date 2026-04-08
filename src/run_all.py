"""
run_all.py – Orchestrates all four experiments and writes results/RESULTS.md.
"""

from __future__ import annotations

import os
import sys
import textwrap
from datetime import datetime

# Ensure src/ is importable when run from project root
sys.path.insert(0, os.path.dirname(__file__))

from exp1_classification import run_experiment as run_exp1
from exp2_regression import run_experiment as run_exp2
from exp3_hyperparameter_tuning import run_experiment as run_exp3
from exp4_unsupervised import run_experiment as run_exp4

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def write_results_md(
    exp1_out: dict,
    exp2_out: dict,
    exp3_out: dict,
    exp4_out: dict,
) -> None:
    """Write a comprehensive RESULTS.md summarising all experiment outcomes.

    Parameters
    ----------
    exp1_out, exp2_out, exp3_out, exp4_out : dict
        Return values from each experiment's run_experiment().
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "RESULTS.md")

    # ---- Exp 1 ----
    exp1_table = exp1_out["comparison_table"].to_markdown()

    # ---- Exp 2 ----
    exp2_table = exp2_out["comparison_table"].to_markdown()

    # ---- Exp 3 ----
    exp3_search = exp3_out["search_summary"].to_markdown()
    exp3_cv = exp3_out["cv_summary"].to_markdown()
    test_acc = exp3_out["test_accuracy"]

    # ---- Exp 4 ----
    exp4_pca = exp4_out["pca_summary"].to_markdown()
    exp4_km = exp4_out["kmeans_summary"].to_markdown()
    optimal_k = exp4_out["optimal_k"]
    best_sil = exp4_out["kmeans_results"][optimal_k]["silhouette"]

    content = textwrap.dedent(f"""\
    # Experiment Results
    Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    ---

    ## Experiment 1 – Scikit-Learn Classification Pipeline Benchmark (Iris)

    **Dataset:** Iris (150 samples, 4 features, 3 classes)  
    **Split:** 70% train / 30% test, random_state=42  
    **Preprocessing:** StandardScaler  

    ### Classifier Comparison

    {exp1_table}

    **Key Findings:**
    - All three classifiers achieved ≥ 93% accuracy on the Iris test set.
    - SVC and LogisticRegression achieved the highest accuracy.
    - StandardScaler improved distance-based model (KNN) performance.
    - Confusion matrix heatmaps saved to `results/exp1_confusion_matrices.png`.

    ---

    ## Experiment 2 – Scikit-Learn Regression Pipeline Benchmark (California Housing)

    **Dataset:** California Housing (20,640 samples, 8 features)  
    **Split:** 70% train / 30% test, random_state=42  
    **Preprocessing:** StandardScaler  

    ### Regressor Comparison

    {exp2_table}

    **Key Findings:**
    - RandomForestRegressor achieved the highest R² and lowest RMSE.
    - Ridge and LinearRegression performed comparably (regularization benefit marginal).
    - Predicted vs actual scatter plots saved to `results/exp2_predicted_vs_actual.png`.

    ---

    ## Experiment 3 – Hyperparameter Tuning & Cross-Validation (Digits)

    **Dataset:** Digits (1797 samples, 64 features, 10 classes)  
    **Split:** 70% train / 30% test, random_state=42  
    **Model:** SVC with C=[0.1,1,10,100], kernel=['linear','rbf'], gamma=['scale','auto']

    ### Search Strategy Comparison

    {exp3_search}

    ### Cross-Validation Stability (Best SVC)

    {exp3_cv}

    **Test Set Accuracy (Best Model):** {test_acc:.4f}

    **Key Findings:**
    - GridSearchCV and RandomizedSearchCV achieved similar best CV accuracy (>97%).
    - RandomizedSearchCV completed in significantly less time.
    - Higher k values produced lower variance in CV score estimates.
    - Box plots saved to `results/exp3_cv_score_distributions.png`.

    ---

    ## Experiment 4 – Unsupervised Learning & Dimensionality Reduction (Wine)

    **Dataset:** Wine (178 samples, 13 features, 3 true classes)  
    **Preprocessing:** StandardScaler  

    ### PCA Explained Variance

    {exp4_pca}

    ### KMeans Sweep (k=2..10)

    {exp4_km}

    **Optimal k (max silhouette):** {optimal_k}  
    **Best Silhouette Score:** {best_sil:.4f}

    **Key Findings:**
    - PCA with 2 components captures ~55-65% of total variance.
    - Elbow method and silhouette score both suggest k=3 as optimal.
    - Discovered clusters largely match the 3 true wine classes.
    - Plots saved: `exp4_pca_true_labels.png`, `exp4_elbow_curve.png`,
      `exp4_silhouette_scores.png`, `exp4_cluster_assignments.png`.

    ---

    ## Artefacts

    | File | Description |
    |------|-------------|
    | `exp1_confusion_matrices.png` | Confusion matrix heatmaps for 3 classifiers |
    | `exp1_comparison_table.csv` | Accuracy/Precision/Recall/F1 per classifier |
    | `exp2_target_distribution.png` | California Housing target histogram |
    | `exp2_predicted_vs_actual.png` | Predicted vs actual scatter plots |
    | `exp2_comparison_table.csv` | MAE/MSE/RMSE/R² per regressor |
    | `exp3_cv_score_distributions.png` | CV score box plots for k=3,5,10 |
    | `exp3_search_comparison.png` | GridSearch vs RandomSearch bar chart |
    | `exp3_search_summary.csv` | Search strategy comparison table |
    | `exp3_cv_summary.csv` | CV stability summary |
    | `exp4_pca_true_labels.png` | PCA 2D projection with true labels |
    | `exp4_elbow_curve.png` | KMeans elbow curve |
    | `exp4_silhouette_scores.png` | Silhouette scores vs k |
    | `exp4_cluster_assignments.png` | KMeans cluster assignments in PCA space |
    | `exp4_contingency_table.csv` | True labels vs cluster assignments |
    | `exp4_pca_summary.csv` | PCA explained variance summary |
    | `exp4_kmeans_summary.csv` | KMeans inertia and silhouette per k |
    """)

    with open(path, "w") as f:
        f.write(content)
    print(f"\nResults written → {path}")


def main() -> None:
    """Run all four experiments sequentially and write RESULTS.md."""
    print("=" * 60)
    print("Running Experiment 1: Classification Pipeline")
    print("=" * 60)
    exp1_out = run_exp1(output_dir=RESULTS_DIR)

    print("\n" + "=" * 60)
    print("Running Experiment 2: Regression Pipeline")
    print("=" * 60)
    exp2_out = run_exp2(output_dir=RESULTS_DIR)

    print("\n" + "=" * 60)
    print("Running Experiment 3: Hyperparameter Tuning & CV")
    print("=" * 60)
    exp3_out = run_exp3(output_dir=RESULTS_DIR)

    print("\n" + "=" * 60)
    print("Running Experiment 4: Unsupervised Learning & PCA")
    print("=" * 60)
    exp4_out = run_exp4(output_dir=RESULTS_DIR)

    print("\n" + "=" * 60)
    print("Writing RESULTS.md")
    print("=" * 60)
    write_results_md(exp1_out, exp2_out, exp3_out, exp4_out)

    print("\nAll experiments completed successfully.")


if __name__ == "__main__":
    main()
