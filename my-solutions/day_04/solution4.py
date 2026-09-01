from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from itertools import product
from sklearn.base import clone
import sklearn.model_selection as model_selection


# This class is just a quick way for me to test all my functionality without loading the data all the time
# It's in no way a good practice, it's just for my own sake. I only use the load_custom data right now 
class Datasets:
    def __init__(self, dataset_name, split_method):
        self.dataset_name = dataset_name
        self.split_method = split_method
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.num_cols = None
        self.cat_cols = None
    
    def load_custom_data(self, features,file_path='data/'):
        training_data = pd.read_csv(file_path + "training_dataset.csv")
        training_data = training_data[features + ['target']]
        testing_data = pd.read_csv(file_path + "testing_dataset.csv")
        testing_data = testing_data[features + ['target']]
        self.X = pd.concat([training_data.drop(columns=['target']), testing_data.drop(columns=['target'])], axis=0)
        self.y = pd.concat([training_data['target'], testing_data['target']], axis=0)
        self.X_train = training_data.drop(columns=['target'])
        self.y_train = training_data['target']
        self.X_test = testing_data.drop(columns=['target'])
        self.y_test = testing_data['target']
          

class KfoldCV:
    def __init__(self, n_samples, k, seed, shuffle=True):
        self.n_samples = n_samples
        self.k = k
        self.seed = seed
        self.shuffle = shuffle
    
    def split(self, X, y):
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(self.n_samples) if self.shuffle else np.arange(self.n_samples)
        fold_sizes = np.full(self.k, self.n_samples // self.k, dtype=int)
        fold_sizes[:self.n_samples % self.k] += 1
        current = 0
        for fold_size in fold_sizes:
            val_idx = indices[current:current + fold_size]
            train_idx = np.concatenate([indices[:current], indices[current + fold_size:]])
            yield train_idx, val_idx
            current += fold_size



# Code from: https://github.com/itdxer/adult-dataset-analysis/blob/master/Classification.ipynb
# The code isn't the same but some of the functionality was inspired and reformatted to fit my structure    
class StratifiedKfoldCV:
    def __init__(self, n_samples, k, seed, shuffle=True):
        self.n_samples = n_samples
        self.k = k
        self.seed = seed
        self.shuffle = shuffle
    
    def split(self, X, y):
        np.random.seed(self.seed)
        # perseve class ratio in each fold, class 1: 37155, class 0: 11687, total: 48842 -> 70/30
        y = np.asarray(y)
        folds = [[] for _ in range(self.k)]
        
        # For flexibility, we will use a dictionary to store the indices of each class
        classes = np.unique(y)
        class_indices = {c: np.where(y == c)[0] for c in classes}
        rng = np.random.default_rng(self.seed)
        if self.shuffle:
            for c in classes:
                rng.shuffle(class_indices[c])
                indices = class_indices[c]
                for i, idx in enumerate(indices):
                    folds[i % self.k].append(idx)

        
        folds = [np.array(fold) for fold in folds]
        
        fold_indices = np.arange(self.k)
        current = 0
        for fold_size in [len(fold) for fold in folds]:
            val_idx = folds[current]
            train_idx = np.concatenate([folds[i] for i in fold_indices if i != current])
            yield train_idx, val_idx
            current += 1
    
def get_metrics(y_true, y_pred, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "fpr": fpr,
        "tpr": tpr,
    }
    return metrics

def print_last_fold_stats(fold_metrics):
    print("Last Fold Metrics:")
    for metric, value in fold_metrics.items():
        print(f"{metric}: {value:.4f}")

def cross_val_score(model, X, y, cv, metrics=["accuracy", "precision", "recall", "f1", "roc_auc"], plot_roc=False):
    scores = defaultdict(list)
    # Iterate through each fold
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = clone(model)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        
        scores_fold = get_metrics(y_val, y_pred, y_pred_proba)
        
        
        
        # This code is a quick and ugly way for testing the values quickly 
        if plot_roc:
            plt.figure(figsize=(8, 6))
            plt.plot(scores_fold["fpr"], scores_fold["tpr"], label=f'ROC curve (area = {scores_fold["roc_auc"]:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.show()
            
        for metric in metrics:
            scores[metric].append(scores_fold[metric])
        
        
    mean_scores = {metric: np.mean(scores[metric]) for metric in metrics if metric not in ["fpr", "tpr"]}
    std_scores = {metric: np.std(scores[metric]) for metric in metrics if metric not in ["fpr", "tpr"]}
    
    return mean_scores, std_scores, scores


class GridSearchCV:
    def __init__(self, model, param_grid, cv):
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.cv_results_ = []
        self.best_model_ = None
    
    def _get_combinations(self, param_grid):
        """
        Generate all combinations of hyperparameters from the param_grid.
        """
        keys, values = zip(*param_grid.items())
        for v in product(*values):
            yield dict(zip(keys, v))
            
            
    def fit(self, X, y):
        
        for params in self._get_combinations(self.param_grid):
            model = clone(self.model)
            model.set_params(**params)
            
            
            # The cloned model is recloned inside cross_val aswell. 
            mean_scores, std_scores, scores = cross_val_score(model, X, y, cv=self.cv)
            self.cv_results_.append({
                "params": params,
                "mean_scores": mean_scores,
                "std_scores": std_scores,
                "scores": scores
            })
            
            # Assuming we are optimizing for accuracy
            if mean_scores["accuracy"] > self.best_score_:
                self.best_score_ = mean_scores["accuracy"]
                self.best_params_ = params
                self.best_model_ = clone(self.model).set_params(**params)
        
        self.best_model_.fit(X, y)

    def predict(self, X):
        return self.best_model_.predict(X)
    


def evaluate_models_cross_val(X, y,cv):
    log_reg = LogisticRegression(max_iter=2000)
    dtc = DecisionTreeClassifier()
    knc = KNeighborsClassifier()
    models = {
        "Logistic Regression": log_reg,
        "Decision Tree Classifier": dtc,
        "K-Nearest Neighbors": knc
    }
    
    results = {}
    
    for model_name, model in models.items():
        mean_scores, std_scores, scores = cross_val_score(model, X, y, cv=cv)
        results[model_name] = {
            "mean_scores": mean_scores,
            "std_scores": std_scores
        }
    return results
        
        
def log_reg_pipeline(case: Datasets,cv, run_grid_search=False):
    log_reg = LogisticRegression(max_iter=2000)
    param_grid = {
        'C': [1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,16.5,17],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']  # 'liblinear' supports both l1 and l2 penalties
    }
    
    
    
    
    X_train, y_train = case.X_train, case.y_train
    X_test, y_test = case.X_test, case.y_test
        
    
    if run_grid_search:
        # Using cross_val_score to evaluate the model with cross-validation
        
        # Using cross_val with grid search to find the best hyperparameters
        grid_search = GridSearchCV(log_reg, param_grid, cv=cv)
        grid_search.fit(X_train, y_train)
        
        model_params = grid_search.best_params_
        model = grid_search.best_model_
        print("Best parameters found: ", grid_search.best_params_)
        return model
    
    else:
        return log_reg
   
def decision_tree_pipeline(case: Datasets,cv, run_grid_search=False):
    dtc = DecisionTreeClassifier()
    param_grid = {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }
    
    X_train, y_train = case.X_train, case.y_train
    X_test, y_test = case.X_test, case.y_test
            
    if run_grid_search:
    # Using cross_val_score to evaluate the model with cross-validation
    
    # Using cross_val with grid search to find the best hyperparameters
        grid_search = GridSearchCV(dtc, param_grid, cv=cv)
        grid_search.fit(X_train, y_train)
        
        model_params = grid_search.best_params_
        print("Best parameters found: ", grid_search.best_params_)
        return grid_search.best_model_
    
    else:
        return dtc
    

def knn_pipeline(case: Datasets, cv, run_grid_search=False):
    knc = KNeighborsClassifier()
    param_grid = {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    }
    
    X_train, y_train = case.X_train, case.y_train
    X_test, y_test = case.X_test, case.y_test
            
    if run_grid_search:
    # Using cross_val_score to evaluate the model with cross-validation
    
    # Using cross_val with grid search to find the best hyperparameters
        grid_search = GridSearchCV(knc, param_grid, cv=cv)
        grid_search.fit(X_train, y_train)
        
        model_params = grid_search.best_params_
        print("Best parameters found: ", grid_search.best_params_)
        return grid_search.best_model_
    
    else:
        return knc

def eval_models(case: Datasets):
   
    X_train_dtc, y_train_dtc = case.X_train, case.y_train
    X_train_knn, y_train_knn = case.X_train, case.y_train
    X_train_log, y_train_log = case.X_train, case.y_train
    
    #cv = model_selection.StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    cv = StratifiedKfoldCV(n_samples=len(X_train_dtc), k=5, seed=42)
    #cv = KfoldCV(n_samples=len(X_train_dtc), k=5, seed=42)
    #cv = model_selection.KFold(n_splits=5, random_state=42, shuffle=True)
    model = decision_tree_pipeline(case, cv, run_grid_search=True)
    knn = knn_pipeline(case, cv, run_grid_search=True)
    log_reg = log_reg_pipeline(case, cv, run_grid_search=True)
    cv_means_dct, cv_stds_dct, cv_scores_dct = cross_val_score(model, X_train_dtc, y_train_dtc, cv=cv, metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc','fpr','tpr'])
    cv_means_knn, cv_stds_knn, cv_scores_knn = cross_val_score(knn, X_train_knn, y_train_knn, cv=cv, metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc','fpr','tpr'])
    cv_means_log, cv_stds_log, cv_scores_log = cross_val_score(log_reg, X_train_log, y_train_log, cv=cv, metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc','fpr','tpr'])


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
    
if __name__ == "__main__":
    case = Datasets(dataset_name="custom", split_method="train_test_split")
    case.load_custom_data()
    eval_models(case)
