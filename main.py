"""
Main entry point for the Scikit-Learn Classification Pipeline Benchmark (exp_1).

Run:
    python main.py
"""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline

from src.pipeline import build_pipeline, run_experiment
from src.reporting import save_results_markdown
from src.visualization import (
    plot_comparison_table,
    plot_confusion_matrix,
    plot_cv_scores,
    plot_f1_ranking,
    plot_gridsearch_heatmap,
)


def main() -> None:
    """Execute the full benchmark experiment and persist all outputs."""
    print("=" * 60)
    print("Scikit-Learn Classification Pipeline Benchmark (exp_1)")
    print("=" * 60)

    # ── Run experiment ────────────────────────────────────────────
    experiment = run_experiment()
    results = experiment["results"]
    comparison_table = experiment["comparison_table"]
    grid_search = experiment["grid_search"]
    X_test = experiment["X_test"]
    y_test = experiment["y_test"]

    # ── Print comparison table ────────────────────────────────────
    print("\n── Classifier Comparison Table (ranked by F1) ──")
    print(comparison_table.to_string())

    # ── Print GridSearchCV results ────────────────────────────────
    print("\n── GridSearchCV (SVC) ──")
    print(f"  Best params : {grid_search.best_params_}")
    print(f"  Best CV acc : {grid_search.best_score_:.4f}")

    # ── Print best model classification report ────────────────────
    best_result = max(results, key=lambda r: r.f1)
    print(f"\n── Best Classifier: {best_result.name} (F1={best_result.f1:.4f}) ──")
    print(best_result.classification_report_str)

    # ── Predictions for confusion matrix (best classifier) ────────
    best_clf_name = best_result.name
    # Re-predict using the best pipeline for the confusion matrix
    if "GridSearchCV" in best_clf_name:
        y_pred_best = grid_search.predict(X_test)
    else:
        from src.pipeline import get_classifiers, split_data, load_dataset
        from sklearn.datasets import load_breast_cancer
        X, y, _ = load_dataset()
        X_train, X_test_local, y_train, y_test_local = split_data(X, y)
        clf_map = get_classifiers()
        clf_name_map = {
            "KNeighborsClassifier": "KNeighborsClassifier",
            "SVC": "SVC",
            "GaussianNB": "GaussianNB",
            "RandomForestClassifier": "RandomForestClassifier",
        }
        clf = clf_map[best_clf_name]
        pipe: Pipeline = build_pipeline(clf)
        pipe.fit(X_train, y_train)
        y_pred_best = pipe.predict(X_test)

    # ── Visualizations ────────────────────────────────────────────
    plot_comparison_table(comparison_table)
    plot_cv_scores(comparison_table)
    plot_confusion_matrix(y_test, y_pred_best, title=f"Confusion Matrix — {best_clf_name}")
    plot_gridsearch_heatmap(grid_search)
    plot_f1_ranking(comparison_table)

    # ── Save RESULTS.md ───────────────────────────────────────────
    save_results_markdown(
        comparison_table=comparison_table,
        results=results,
        best_params=grid_search.best_params_,
        best_cv_score=grid_search.best_score_,
    )

    print("\n✓ All results saved to results/")


if __name__ == "__main__":
    main()
