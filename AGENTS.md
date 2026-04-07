# Repository Knowledge — qca-scikit-learn-classification-pipeline-benchmark

## Project Purpose
Scikit-learn classification pipeline benchmark (exp_1).
Compares KNN, SVC, GaussianNB, RandomForest on breast_cancer dataset.

## Key Design Decisions
- All classifiers wrapped in `sklearn.pipeline.Pipeline` with `StandardScaler` to prevent data leakage
- `random_state=42` used everywhere for reproducibility
- `StratifiedKFold(n_splits=5)` for cross-validation
- GridSearchCV on SVC: `C=[0.1,1,10]`, `kernel=['linear','rbf']`
- Results saved to `results/RESULTS.md` and PNG figures

## Module Map
- `src/pipeline.py` — core experiment logic (load, split, evaluate, grid search)
- `src/visualization.py` — matplotlib/seaborn plots
- `src/reporting.py` — RESULTS.md generation
- `main.py` — orchestrates full experiment
- `tests/test_pipeline.py` — methodology adherence + metric range tests
- `tests/test_visualization.py` — plot file generation tests

## Library Choices (Context7 verified)
- scikit-learn Pipeline, GridSearchCV, cross_val_score — standard sklearn API
- matplotlib + seaborn for visualization
- tabulate for markdown table rendering

## Constants
- RANDOM_STATE = 42
- TEST_SIZE = 0.2
- CV_FOLDS = 5
