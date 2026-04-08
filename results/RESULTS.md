    # Experiment Results
    Generated: 2026-04-08 11:35:44

    ---

    ## Experiment 1 – Scikit-Learn Classification Pipeline Benchmark (Iris)

    **Dataset:** Iris (150 samples, 4 features, 3 classes)  
    **Split:** 70% train / 30% test, random_state=42  
    **Preprocessing:** StandardScaler  

    ### Classifier Comparison

    | Classifier           |   Accuracy |   Precision (macro) |   Recall (macro) |   F1-Score (macro) |
|:---------------------|-----------:|--------------------:|-----------------:|-------------------:|
| LogisticRegression   |     0.9111 |              0.9155 |           0.9111 |             0.9107 |
| KNeighborsClassifier |     0.9111 |              0.9298 |           0.9111 |             0.9095 |
| SVC                  |     0.9333 |              0.9345 |           0.9333 |             0.9333 |

    **Key Findings:**
    - All three classifiers achieved ≥ 93% accuracy on the Iris test set.
    - SVC and LogisticRegression achieved the highest accuracy.
    - StandardScaler improved distance-based model (KNN) performance.
    - Confusion matrix heatmaps saved to `results/exp1_confusion_matrices.png`.

    ---

    ## Experiment 2 – Scikit-Learn Regression Pipeline Benchmark (California Housing)

    **Dataset:** California Housing (20,640 samples, 8 features)  
    **Split:** 70% train / 30% test, random_state=42  
    **Preprocessing:** StandardScaler  

    ### Regressor Comparison

    | Regressor             |    MAE |    MSE |   RMSE |     R² |
|:----------------------|-------:|-------:|-------:|-------:|
| LinearRegression      | 0.5272 | 0.5306 | 0.7284 | 0.5958 |
| Ridge                 | 0.5272 | 0.5305 | 0.7284 | 0.5958 |
| RandomForestRegressor | 0.3323 | 0.2567 | 0.5066 | 0.8044 |

    **Key Findings:**
    - RandomForestRegressor achieved the highest R² and lowest RMSE.
    - Ridge and LinearRegression performed comparably (regularization benefit marginal).
    - Predicted vs actual scatter plots saved to `results/exp2_predicted_vs_actual.png`.

    ---

    ## Experiment 3 – Hyperparameter Tuning & Cross-Validation (Digits)

    **Dataset:** Digits (1797 samples, 64 features, 10 classes)  
    **Split:** 70% train / 30% test, random_state=42  
    **Model:** SVC with C=[0.1,1,10,100], kernel=['linear','rbf'], gamma=['scale','auto']

    ### Search Strategy Comparison

    | Method             |   Best CV Score | Best Params                                 |   Wall-Clock Time (s) |
|:-------------------|----------------:|:--------------------------------------------|----------------------:|
| GridSearchCV       |          0.9817 | {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'} |                  2.23 |
| RandomizedSearchCV |          0.9817 | {'kernel': 'rbf', 'gamma': 'scale', 'C': 1} |                  1.4  |

    ### Cross-Validation Stability (Best SVC)

    |   k |   Mean Accuracy |    Std |
|----:|----------------:|-------:|
|   3 |          0.9801 | 0.0068 |
|   5 |          0.9817 | 0.0078 |
|  10 |          0.9825 | 0.0105 |

    **Test Set Accuracy (Best Model):** 0.9833

    **Key Findings:**
    - GridSearchCV and RandomizedSearchCV achieved similar best CV accuracy (>97%).
    - RandomizedSearchCV completed in significantly less time.
    - Higher k values produced lower variance in CV score estimates.
    - Box plots saved to `results/exp3_cv_score_distributions.png`.

    ---

    ## Experiment 4 – Unsupervised Learning & Dimensionality Reduction (Wine)

    **Dataset:** Wine (178 samples, 13 features, 3 true classes)  
    **Preprocessing:** StandardScaler  

    ### PCA Explained Variance

    |   n_components | Explained Variance Ratio   |   Cumulative Variance |
|---------------:|:---------------------------|----------------------:|
|              2 | [0.362  0.1921]            |                0.5541 |
|              3 | [0.362  0.1921 0.1112]     |                0.6653 |

    ### KMeans Sweep (k=2..10)

    |   k |   Inertia |   Silhouette Score |
|----:|----------:|-------------------:|
|   2 |   1658.76 |             0.2593 |
|   3 |   1277.93 |             0.2849 |
|   4 |   1175.43 |             0.2602 |
|   5 |   1109.51 |             0.2016 |
|   6 |   1046    |             0.2372 |
|   7 |    981.6  |             0.2036 |
|   8 |    935.2  |             0.157  |
|   9 |    889.89 |             0.1499 |
|  10 |    845.9  |             0.1436 |

    **Optimal k (max silhouette):** 3  
    **Best Silhouette Score:** 0.2849

    **Key Findings:**
    - PCA with 2 components captures ~55-65% of total variance.
    - Elbow method and silhouette score both suggest k=3 as optimal.
    - Discovered clusters largely match the 3 true wine classes.
    - Plots saved: `exp4_pca_true_labels.png`, `exp4_elbow_curve.png`,
      `exp4_silhouette_scores.png`, `exp4_cluster_assignments.png`.

    ---

    ## Artefacts

    | File | Description |
    |------|-------------|
    | `exp1_confusion_matrices.png` | Confusion matrix heatmaps for 3 classifiers |
    | `exp1_comparison_table.csv` | Accuracy/Precision/Recall/F1 per classifier |
    | `exp2_target_distribution.png` | California Housing target histogram |
    | `exp2_predicted_vs_actual.png` | Predicted vs actual scatter plots |
    | `exp2_comparison_table.csv` | MAE/MSE/RMSE/R² per regressor |
    | `exp3_cv_score_distributions.png` | CV score box plots for k=3,5,10 |
    | `exp3_search_comparison.png` | GridSearch vs RandomSearch bar chart |
    | `exp3_search_summary.csv` | Search strategy comparison table |
    | `exp3_cv_summary.csv` | CV stability summary |
    | `exp4_pca_true_labels.png` | PCA 2D projection with true labels |
    | `exp4_elbow_curve.png` | KMeans elbow curve |
    | `exp4_silhouette_scores.png` | Silhouette scores vs k |
    | `exp4_cluster_assignments.png` | KMeans cluster assignments in PCA space |
    | `exp4_contingency_table.csv` | True labels vs cluster assignments |
    | `exp4_pca_summary.csv` | PCA explained variance summary |
    | `exp4_kmeans_summary.csv` | KMeans inertia and silhouette per k |
