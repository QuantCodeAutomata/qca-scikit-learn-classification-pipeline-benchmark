# Scikit-Learn Classification Pipeline Benchmark — Results
## Experiment: exp_1
**Dataset:** Breast Cancer Wisconsin (sklearn.datasets.load_breast_cancer)
**Split:** 80% train / 20% test, stratified, random_state=42
**Cross-Validation:** 5-fold Stratified K-Fold on training set

---
## Classifier Comparison Table (ranked by Test F1-Score)
|    | Classifier              |   Test Accuracy |   Test Precision |   Test Recall |   Test F1 |   CV Mean Accuracy |   CV Std |
|---:|:------------------------|----------------:|-----------------:|--------------:|----------:|-------------------:|---------:|
|  1 | SVC                     |          0.9825 |           0.9825 |        0.9825 |    0.9825 |             0.9692 |   0.0146 |
|  2 | SVC (GridSearchCV best) |          0.9825 |           0.9825 |        0.9825 |    0.9825 |             0.9758 |   0      |
|  3 | KNeighborsClassifier    |          0.9561 |           0.9561 |        0.9561 |    0.956  |             0.9626 |   0.0112 |
|  4 | RandomForestClassifier  |          0.9561 |           0.9561 |        0.9561 |    0.956  |             0.9626 |   0.0179 |
|  5 | GaussianNB              |          0.9298 |           0.9298 |        0.9298 |    0.9298 |             0.9341 |   0.0287 |

---
## GridSearchCV — SVC Hyperparameter Tuning
- **Param grid:** `C=[0.1, 1, 10]`, `kernel=['linear', 'rbf']`
- **Best params:** `{'clf__C': 0.1, 'clf__kernel': 'linear'}`
- **Best CV accuracy:** `0.9758`

---
## Per-Classifier Classification Reports
### KNeighborsClassifier
```
              precision    recall  f1-score   support

           0       0.95      0.93      0.94        42
           1       0.96      0.97      0.97        72

    accuracy                           0.96       114
   macro avg       0.96      0.95      0.95       114
weighted avg       0.96      0.96      0.96       114

```
### SVC
```
              precision    recall  f1-score   support

           0       0.98      0.98      0.98        42
           1       0.99      0.99      0.99        72

    accuracy                           0.98       114
   macro avg       0.98      0.98      0.98       114
weighted avg       0.98      0.98      0.98       114

```
### GaussianNB
```
              precision    recall  f1-score   support

           0       0.90      0.90      0.90        42
           1       0.94      0.94      0.94        72

    accuracy                           0.93       114
   macro avg       0.92      0.92      0.92       114
weighted avg       0.93      0.93      0.93       114

```
### RandomForestClassifier
```
              precision    recall  f1-score   support

           0       0.95      0.93      0.94        42
           1       0.96      0.97      0.97        72

    accuracy                           0.96       114
   macro avg       0.96      0.95      0.95       114
weighted avg       0.96      0.96      0.96       114

```
### SVC (GridSearchCV best)
```
              precision    recall  f1-score   support

           0       0.98      0.98      0.98        42
           1       0.99      0.99      0.99        72

    accuracy                           0.98       114
   macro avg       0.98      0.98      0.98       114
weighted avg       0.98      0.98      0.98       114

```

---
## Figures
| File | Description |
|------|-------------|
| `classifier_comparison.png` | Bar chart of test metrics per classifier |
| `cv_scores.png` | Cross-validation accuracy with error bars |
| `confusion_matrix_best.png` | Confusion matrix for best classifier |
| `gridsearch_heatmap.png` | GridSearchCV SVC accuracy heatmap |
| `f1_ranking.png` | Horizontal bar chart ranked by F1-score |
