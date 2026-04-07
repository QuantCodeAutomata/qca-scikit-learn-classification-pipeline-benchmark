# Scikit-Learn Classification Pipeline Benchmark

A reproducible multi-classifier benchmark experiment (exp_1) using scikit-learn Pipelines.

## Overview

Compares KNeighborsClassifier, SVC, GaussianNB, and RandomForestClassifier on the
Breast Cancer Wisconsin dataset using:
- Stratified 80/20 train/test split (`random_state=42`)
- `StandardScaler` inside a `Pipeline` (no data leakage)
- 5-fold stratified cross-validation
- `GridSearchCV` on SVC (`C=[0.1,1,10]`, `kernel=['linear','rbf']`)

## Project Structure

```
.
├── main.py                  # Experiment entry point
├── src/
│   ├── pipeline.py          # Core experiment logic
│   ├── visualization.py     # All plots
│   └── reporting.py         # RESULTS.md generation
├── tests/
│   ├── test_pipeline.py     # Pipeline & methodology tests
│   └── test_visualization.py
├── results/                 # Generated outputs (figures + RESULTS.md)
└── requirements.txt
```

## Usage

```bash
pip install -r requirements.txt
python main.py
```

## Run Tests

```bash
pytest tests/ -v
```

## Results

All metrics and figures are saved to `results/`. See `results/RESULTS.md` for the
full comparison table and per-classifier classification reports.
