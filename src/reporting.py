"""
Reporting Module
================
Saves experiment metrics and results to results/RESULTS.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.pipeline import ClassifierResult

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_results_markdown(
    comparison_table: pd.DataFrame,
    results: list[ClassifierResult],
    best_params: dict,
    best_cv_score: float,
) -> None:
    """Write all experiment metrics to results/RESULTS.md.

    Parameters
    ----------
    comparison_table : ranked DataFrame of classifier metrics
    results : list of ClassifierResult objects
    best_params : best hyperparameters from GridSearchCV
    best_cv_score : best CV accuracy from GridSearchCV
    """
    md_path = RESULTS_DIR / "RESULTS.md"

    lines = [
        "# Scikit-Learn Classification Pipeline Benchmark — Results\n",
        "## Experiment: exp_1\n",
        "**Dataset:** Breast Cancer Wisconsin (sklearn.datasets.load_breast_cancer)\n",
        "**Split:** 80% train / 20% test, stratified, random_state=42\n",
        "**Cross-Validation:** 5-fold Stratified K-Fold on training set\n",
        "\n---\n",
        "## Classifier Comparison Table (ranked by Test F1-Score)\n",
        comparison_table.to_markdown(index=True),
        "\n\n---\n",
        "## GridSearchCV — SVC Hyperparameter Tuning\n",
        f"- **Param grid:** `C=[0.1, 1, 10]`, `kernel=['linear', 'rbf']`\n",
        f"- **Best params:** `{best_params}`\n",
        f"- **Best CV accuracy:** `{best_cv_score:.4f}`\n",
        "\n---\n",
        "## Per-Classifier Classification Reports\n",
    ]

    for r in results:
        lines.append(f"### {r.name}\n")
        lines.append(f"```\n{r.classification_report_str}\n```\n")

    lines += [
        "\n---\n",
        "## Figures\n",
        "| File | Description |\n",
        "|------|-------------|\n",
        "| `classifier_comparison.png` | Bar chart of test metrics per classifier |\n",
        "| `cv_scores.png` | Cross-validation accuracy with error bars |\n",
        "| `confusion_matrix_best.png` | Confusion matrix for best classifier |\n",
        "| `gridsearch_heatmap.png` | GridSearchCV SVC accuracy heatmap |\n",
        "| `f1_ranking.png` | Horizontal bar chart ranked by F1-score |\n",
    ]

    md_path.write_text("".join(lines), encoding="utf-8")
    print(f"Saved: {md_path}")
