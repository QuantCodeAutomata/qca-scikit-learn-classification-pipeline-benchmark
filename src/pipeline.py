"""
Classification Pipeline Module
================================
Implements a reproducible multi-classifier benchmark using scikit-learn Pipelines.

Methodology (exp_1):
1. Load breast_cancer dataset from sklearn.datasets
2. Split 80/20 train/test with random_state=42
3. Wrap StandardScaler + classifier in a Pipeline (prevents data leakage)
4. Evaluate KNN, SVC, GaussianNB, RandomForest on test set
5. 5-fold stratified cross-validation on training set
6. GridSearchCV on SVC with C=[0.1,1,10] x kernel=['linear','rbf']
7. Compile comparison table ranked by F1-score

# Using scikit-learn Pipeline, GridSearchCV, cross_val_score — Context7 confirmed
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


@dataclass
class ClassifierResult:
    """Holds evaluation metrics for a single classifier."""

    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    cv_mean: float
    cv_std: float
    best_params: dict[str, Any] = field(default_factory=dict)
    classification_report_str: str = ""


def load_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the breast cancer benchmark dataset.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    feature_names : list[str]
    """
    data = load_breast_cancer()
    logger.info(
        "Loaded breast_cancer dataset: %d samples, %d features, %d classes",
        data.data.shape[0],
        data.data.shape[1],
        len(data.target_names),
    )
    return data.data, data.target, list(data.feature_names)


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split.

    Parameters
    ----------
    X : feature matrix
    y : target vector
    test_size : fraction of data reserved for testing
    random_state : reproducibility seed

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_pipeline(classifier: Any) -> Pipeline:
    """Wrap StandardScaler and a classifier in a sklearn Pipeline.

    Using sklearn Pipeline — Context7 confirmed (prevents data leakage by
    fitting the scaler only on training data during cross-validation).

    Parameters
    ----------
    classifier : sklearn estimator

    Returns
    -------
    Pipeline with steps [('scaler', StandardScaler()), ('clf', classifier)]
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", classifier),
        ]
    )


def get_classifiers() -> dict[str, Any]:
    """Return the benchmark classifiers as specified in the methodology.

    Returns
    -------
    dict mapping classifier name → sklearn estimator instance
    """
    return {
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=5),
        "SVC": SVC(kernel="rbf", random_state=RANDOM_STATE),
        "GaussianNB": GaussianNB(),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
    }


def evaluate_classifier(
    name: str,
    classifier: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cv: int = CV_FOLDS,
) -> ClassifierResult:
    """Fit a pipeline, evaluate on test set, and run stratified CV.

    Parameters
    ----------
    name : human-readable classifier name
    classifier : sklearn estimator
    X_train, X_test : feature matrices
    y_train, y_test : target vectors
    cv : number of cross-validation folds

    Returns
    -------
    ClassifierResult with all metrics populated
    """
    pipe = build_pipeline(classifier)

    # Fit on training data (scaler fitted only on X_train — no leakage)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_test, y_pred)

    # 5-fold stratified CV on training set
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        build_pipeline(classifier),  # fresh pipeline to avoid state leakage
        X_train,
        y_train,
        cv=skf,
        scoring="accuracy",
    )

    logger.info(
        "%s — test_acc=%.4f  cv_mean=%.4f±%.4f",
        name,
        acc,
        cv_scores.mean(),
        cv_scores.std(),
    )

    return ClassifierResult(
        name=name,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        cv_mean=cv_scores.mean(),
        cv_std=cv_scores.std(),
        classification_report_str=report,
    )


def run_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = CV_FOLDS,
) -> GridSearchCV:
    """Run GridSearchCV on SVC inside a Pipeline.

    Param grid: C=[0.1, 1, 10], kernel=['linear', 'rbf'] as per methodology.

    Parameters
    ----------
    X_train : training feature matrix
    y_train : training target vector
    cv : number of cross-validation folds

    Returns
    -------
    Fitted GridSearchCV object
    """
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", SVC(random_state=RANDOM_STATE)),
        ]
    )

    param_grid = {
        "clf__C": [0.1, 1, 10],
        "clf__kernel": ["linear", "rbf"],
    }

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=skf,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    logger.info(
        "GridSearchCV best params: %s  best_score=%.4f",
        grid_search.best_params_,
        grid_search.best_score_,
    )
    return grid_search


def build_comparison_table(results: list[ClassifierResult]) -> pd.DataFrame:
    """Compile classifier metrics into a ranked comparison DataFrame.

    Parameters
    ----------
    results : list of ClassifierResult objects

    Returns
    -------
    pd.DataFrame sorted descending by F1-score
    """
    rows = [
        {
            "Classifier": r.name,
            "Test Accuracy": round(r.accuracy, 4),
            "Test Precision": round(r.precision, 4),
            "Test Recall": round(r.recall, 4),
            "Test F1": round(r.f1, 4),
            "CV Mean Accuracy": round(r.cv_mean, 4),
            "CV Std": round(r.cv_std, 4),
        }
        for r in results
    ]
    df = pd.DataFrame(rows).sort_values("Test F1", ascending=False).reset_index(drop=True)
    df.index += 1  # rank starts at 1
    return df


def run_experiment() -> dict[str, Any]:
    """Execute the full classification benchmark experiment.

    Steps follow the methodology exactly:
    1. Load dataset
    2. Train/test split (80/20, stratified, random_state=42)
    3. Evaluate KNN, SVC, GaussianNB, RandomForest via Pipeline
    4. 5-fold stratified CV for each classifier
    5. GridSearchCV on SVC
    6. Build comparison table

    Returns
    -------
    dict with keys: results, comparison_table, grid_search, X_test, y_test
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    X, y, feature_names = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)

    classifiers = get_classifiers()
    results: list[ClassifierResult] = []

    for name, clf in classifiers.items():
        result = evaluate_classifier(name, clf, X_train, X_test, y_train, y_test)
        results.append(result)

    # GridSearchCV on SVC
    grid_search = run_grid_search(X_train, y_train)
    best_svc_params = grid_search.best_params_

    # Evaluate best SVC from GridSearchCV on test set
    y_pred_best = grid_search.predict(X_test)
    best_svc_result = ClassifierResult(
        name="SVC (GridSearchCV best)",
        accuracy=accuracy_score(y_test, y_pred_best),
        precision=precision_score(y_test, y_pred_best, average="weighted", zero_division=0),
        recall=recall_score(y_test, y_pred_best, average="weighted", zero_division=0),
        f1=f1_score(y_test, y_pred_best, average="weighted", zero_division=0),
        cv_mean=grid_search.best_score_,
        cv_std=0.0,
        best_params=best_svc_params,
        classification_report_str=classification_report(y_test, y_pred_best),
    )
    results.append(best_svc_result)

    comparison_table = build_comparison_table(results)

    return {
        "results": results,
        "comparison_table": comparison_table,
        "grid_search": grid_search,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
    }
