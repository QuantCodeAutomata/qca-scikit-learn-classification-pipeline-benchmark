# qca-scikit-learn-classification-pipeline-benchmark

Structured benchmark experiments evaluating Scikit-Learn classification,
regression, and preprocessing pipelines on standardised datasets.

## Experiments

| ID | Title | Dataset |
|----|-------|---------|
| exp_1 | Classification Pipeline Benchmark | Breast Cancer |
| exp_2 | Regression Pipeline Benchmark | Diabetes |
| exp_3 | Preprocessing & Feature Engineering Impact Study | Wine + Synthetic |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments (saves results to results/)
python run_experiments.py

# Run tests
pytest tests/ -v
```

## Repository Structure

```
src/
  exp1_classification.py   # Exp 1: 4 classifiers, GridSearchCV, metrics
  exp2_regression.py       # Exp 2: 4 regressors, GridSearchCV, metrics
  exp3_preprocessing.py    # Exp 3: scaling, split ratio, encoding study
  utils.py                 # Shared plotting and result utilities
tests/
  test_exp1.py
  test_exp2.py
  test_exp3.py
results/
  RESULTS.md               # Aggregated metrics (generated)
  *.png                    # Plots (generated)
run_experiments.py         # Entry point
```

## Key Design Decisions

- All scikit-learn APIs used directly (Context7 confirmed standard library coverage)
- `GridSearchCV` with `n_jobs=-1` for parallelism
- `random_state=42` throughout for reproducibility
- Results automatically saved to `results/` directory
- Tests validate mathematical properties with `assert` statements

## Expected Outcomes

- **Exp 1**: RF and SVC achieve >95% accuracy on breast cancer after tuning
- **Exp 2**: RF outperforms linear models; Ridge/Lasso beat plain LinearRegression
- **Exp 3**: Scaling significantly improves SVC; one-hot encoding outperforms label encoding
