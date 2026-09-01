from ..day_04.solution4 import Datasets, StratifiedKfoldCV, GridSearchCV, cross_val_score, KfoldCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

data = Datasets('custom',split_method='sklearn')
data.load_custom_data(file_path='data/')  

def log_reg_pipeline(case: Datasets):
    log_reg = LogisticRegression(max_iter=2000)
    param_grid = {
        'C': [1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,16.5,17],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']  # 'liblinear' supports both l1 and l2 penalties
    }
    
    
    
    
    X_train, y_train = case.X_train, case.y_train
    X_test, y_test = case.X_test, case.y_test
    
    log_reg.fit(X_train, y_train)
    
    cv = StratifiedKfoldCV(n_samples=len(X_train), k=5, seed=42)
    
    # Using cross_val_score to evaluate the model with cross-validation
    
    # Using cross_val with grid search to find the best hyperparameters
    grid_search = GridSearchCV(log_reg, param_grid, cv=cv)
    grid_search.fit(X_train, y_train)
    
    model_params = grid_search.best_params_
    model = LogisticRegression(**model_params)
    return model
   
def decision_tree_pipeline(case: Datasets):
    dtc = DecisionTreeClassifier()
    param_grid = {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }
    
    X_train, y_train = case.X_train, case.y_train
    X_test, y_test = case.X_test, case.y_test
    
    dtc.fit(X_train, y_train)
    
    cv = StratifiedKfoldCV(n_samples=len(X_train), k=5, seed=42)
    
    # Using cross_val_score to evaluate the model with cross-validation
    
    # Using cross_val with grid search to find the best hyperparameters
    grid_search = GridSearchCV(dtc, param_grid, cv=cv)
    grid_search.fit(X_train, y_train)
    
    model_params = grid_search.best_params_
    model = DecisionTreeClassifier(**model_params)
    return model
    

def knn_pipeline(case: Datasets):
    knc = KNeighborsClassifier()
    param_grid = {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    }
    
    X_train, y_train = case.X_train, case.y_train
    X_test, y_test = case.X_test, case.y_test
    
    knc.fit(X_train, y_train)
    
    cv = StratifiedKfoldCV(n_samples=len(X_train), k=5, seed=42)
    
    # Using cross_val_score to evaluate the model with cross-validation
    
    # Using cross_val with grid search to find the best hyperparameters
    grid_search = GridSearchCV(knc, param_grid, cv=cv)
    grid_search.fit(X_train, y_train)
    
    model_params = grid_search.best_params_
    model = KNeighborsClassifier(**model_params)
    return model

def eval_models(case: Datasets):
   
    X_train_dtc, y_train_dtc = case.X_train, case.y_train
    X_train_knn, y_train_knn = case.X_train, case.y_train
    X_train_log, y_train_log = case.X_train, case.y_train
    cv = KfoldCV(n_samples=len(X_train_dtc), k=5, seed=42)

    
    model = decision_tree_pipeline(case)
    knn = knn_pipeline(case)
    log_reg = log_reg_pipeline(case)
    cv_means_dct, cv_stds_dct, cv_scores_dct = cross_val_score(model, X_train_dtc, y_train_dtc, cv=cv)
    cv_means_knn, cv_stds_knn, cv_scores_knn = cross_val_score(knn, X_train_knn, y_train_knn, cv=cv)
    cv_means_log, cv_stds_log, cv_scores_log = cross_val_score(log_reg, X_train_log, y_train_log, cv=cv)


    print("Mean scores with standard deviation For DecisionTreeClassifier:")
    print("Accuracy: {:.4f} ± {:.4f}".format(cv_means_dct['accuracy'], cv_stds_dct['accuracy']))
    print("Precision: {:.4f} ± {:.4f}".format(cv_means_dct['precision'], cv_stds_dct['precision']))
    print("Recall: {:.4f} ± {:.4f}".format(cv_means_dct['recall'], cv_stds_dct['recall']))
    print("F1 Score: {:.4f} ± {:.4f}".format(cv_means_dct['f1'], cv_stds_dct['f1']))
    print("ROC-AUC: {:.4f} ± {:.4f}".format(cv_means_dct['roc_auc'], cv_stds_dct['roc_auc']))

    print('-' * 30)
    print("Mean scores with standard deviation for KNeighbours:")
    print("Accuracy: {:.4f} ± {:.4f}".format(cv_means_knn['accuracy'], cv_stds_knn['accuracy']))
    print("Precision: {:.4f} ± {:.4f}".format(cv_means_knn['precision'], cv_stds_knn['precision']))
    print("Recall: {:.4f} ± {:.4f}".format(cv_means_knn['recall'], cv_stds_knn['recall']))
    print("F1 Score: {:.4f} ± {:.4f}".format(cv_means_knn['f1'], cv_stds_knn['f1']))
    print("ROC-AUC: {:.4f} ± {:.4f}".format(cv_means_knn['roc_auc'], cv_stds_knn['roc_auc']))
    print('-' * 30)
    print("Mean scores with standard deviation for LogisticRegression:")
    print("Accuracy: {:.4f} ± {:.4f}".format(cv_means_log['accuracy'], cv_stds_log['accuracy']))
    print("Precision: {:.4f} ± {:.4f}".format(cv_means_log['precision'], cv_stds_log['precision']))
    print("Recall: {:.4f} ± {:.4f}".format(cv_means_log['recall'], cv_stds_log['recall']))
    print("F1 Score: {:.4f} ± {:.4f}".format(cv_means_log['f1'], cv_stds_log['f1']))
    print("ROC-AUC: {:.4f} ± {:.4f}".format(cv_means_log['roc_auc'], cv_stds_log['roc_auc']))
