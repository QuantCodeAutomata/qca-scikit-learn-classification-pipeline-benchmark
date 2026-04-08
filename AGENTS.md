# Repository: qca-scikit-learn-classification-pipeline-benchmark

## Overview
Structured benchmark experiments evaluating Scikit-Learn classification, regression,
and preprocessing pipelines on standardized datasets.

## Structure
```
src/
  exp1_classification.py  - Breast cancer classification benchmark (4 classifiers)
  exp2_regression.py      - Diabetes regression benchmark (4 regressors)
  exp3_preprocessing.py   - Wine dataset preprocessing impact study
  utils.py                - Shared utilities (plotting, table formatting)
tests/
  test_exp1.py            - Tests for classification experiment
  test_exp2.py            - Tests for regression experiment
  test_exp3.py            - Tests for preprocessing experiment
results/
  RESULTS.md              - Aggregated metrics and findings
  *.png                   - Generated plots
run_experiments.py        - Entry point: runs all experiments and saves results
```

## Key Design Decisions
- All scikit-learn APIs used directly (Context7 confirmed standard library coverage)
- GridSearchCV with n_jobs=-1 for parallelism
- random_state=42 throughout for reproducibility
- Results saved to results/ directory automatically
- Tests use assert statements to validate mathematical properties

## Datasets
- exp1: sklearn.datasets.load_breast_cancer() — binary classification
- exp2: sklearn.datasets.load_diabetes() — regression
- exp3: sklearn.datasets.load_wine() — multi-class classification

## Running
```bash
python run_experiments.py        # Run all experiments
pytest tests/ -v                 # Run all tests
```
