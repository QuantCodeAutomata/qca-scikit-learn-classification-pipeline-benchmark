"""Main entry point — runs all three experiments and saves results.

Usage
-----
    python run_experiments.py

Outputs
-------
    results/RESULTS.md          — aggregated metrics in Markdown
    results/exp1_*.png          — Experiment 1 plots
    results/exp2_*.png          — Experiment 2 plots
    results/exp3*.png           — Experiment 3 plots
"""

from __future__ import annotations

import sys
import time

from src.utils import init_results_md
from src.exp1_classification import run_experiment_1
from src.exp2_regression import run_experiment_2
from src.exp3_preprocessing import run_experiment_3


def main() -> None:
    """Run all experiments sequentially and report total elapsed time."""
    print("=" * 70)
    print("  Scikit-Learn Classification Pipeline Benchmark — All Experiments")
    print("=" * 70)

    init_results_md()
    t0 = time.time()

    # -----------------------------------------------------------------------
    print("\n>>> EXPERIMENT 1: Classification Pipeline Benchmark")
    t1 = time.time()
    df1 = run_experiment_1()
    print(f"    Completed in {time.time() - t1:.1f}s")

    # -----------------------------------------------------------------------
    print("\n>>> EXPERIMENT 2: Regression Pipeline Benchmark")
    t2 = time.time()
    df2 = run_experiment_2()
    print(f"    Completed in {time.time() - t2:.1f}s")

    # -----------------------------------------------------------------------
    print("\n>>> EXPERIMENT 3: Preprocessing & Feature Engineering Impact Study")
    t3 = time.time()
    results3 = run_experiment_3()
    print(f"    Completed in {time.time() - t3:.1f}s")

    # -----------------------------------------------------------------------
    total = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  All experiments completed in {total:.1f}s")
    print("  Results saved to: results/RESULTS.md")
    print("  Plots saved to:   results/*.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
