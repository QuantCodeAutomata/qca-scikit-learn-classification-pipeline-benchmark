# Scikit-Learn Classification Pipeline Benchmark

A comprehensive benchmark repository implementing four Scikit-Learn experiments:

1. **Exp 1** – Classification Pipeline (Iris dataset): LogisticRegression, KNeighborsClassifier, SVC
2. **Exp 2** – Regression Pipeline (California Housing): LinearRegression, Ridge, RandomForestRegressor
3. **Exp 3** – Hyperparameter Tuning & Cross-Validation (Digits dataset): GridSearchCV vs RandomizedSearchCV
4. **Exp 4** – Unsupervised Learning & Dimensionality Reduction (Wine dataset): PCA + KMeans

## Setup

```bash
pip install -r requirements.txt
```

## Run All Experiments

```bash
python src/run_all.py
```

## Run Tests

```bash
pytest tests/ -v
```

## Results

All metrics and plots are saved in the `results/` directory. See `results/RESULTS.md` for a full summary.
