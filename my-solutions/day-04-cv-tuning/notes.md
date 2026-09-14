


# K folds
https://stackoverflow.com/questions/57021928/how-to-create-a-k-fold-cross-validation-test

Here's how it works: On the first iteration, the data is divided into train and test sets. The test size (unless specified as an argument test_size) defaults to n_samples // (n_splits + 1)) and the train size defaults to i * n_samples // (n_splits + 1) + n_samples % (n_splits + 1) where i=current split. As the current split increases, i increases, and the train size increases.



# Running No Gridsearch, and KFold
Mean scores with standard deviation For DecisionTreeClassifier:
Accuracy: 0.7852 ± 0.0018
Precision: 0.5563 ± 0.0164
Recall: 0.5058 ± 0.0072
F1 Score: 0.5297 ± 0.0091
ROC-AUC: 0.7442 ± 0.0040
------------------------------
Mean scores with standard deviation for KNeighbours:
Accuracy: 0.8119 ± 0.0034
Precision: 0.6193 ± 0.0150
Recall: 0.5553 ± 0.0119
F1 Score: 0.5854 ± 0.0106
ROC-AUC: 0.8308 ± 0.0037
------------------------------
Mean scores with standard deviation for LogisticRegression:
Accuracy: 0.8268 ± 0.0021
Precision: 0.6803 ± 0.0177
Recall: 0.5208 ± 0.0069
F1 Score: 0.5899 ± 0.0102
ROC-AUC: 0.8779 ± 0.0024

# GridSearchCV with KFold from sklearn
- Gridsearch: Custom
- Fold: sklearn
- Cross-val: Custom

Mean scores with standard deviation For DecisionTreeClassifier:
Accuracy: 0.8255 ± 0.0036
Precision: 0.6574 ± 0.0144
Recall: 0.5659 ± 0.0135
F1 Score: 0.6080 ± 0.0091
ROC-AUC: 0.8607 ± 0.0037
------------------------------
Mean scores with standard deviation for KNeighbours:
Accuracy: 0.8222 ± 0.0053
Precision: 0.6484 ± 0.0144
Recall: 0.5610 ± 0.0129
F1 Score: 0.6015 ± 0.0128
ROC-AUC: 0.8567 ± 0.0067
------------------------------
Mean scores with standard deviation for LogisticRegression:
Accuracy: 0.8268 ± 0.0047
Precision: 0.6802 ± 0.0145
Recall: 0.5209 ± 0.0112
F1 Score: 0.5900 ± 0.0122
ROC-AUC: 0.8779 ± 0.0048

# GridSearch with custom KFold
- GS: Custom
- KFold: Custom
- CV: Custom

Mean scores with standard deviation For DecisionTreeClassifier:
Accuracy: 0.8266 ± 0.0018
Precision: 0.6665 ± 0.0200
Recall: 0.5521 ± 0.0217
F1 Score: 0.6034 ± 0.0109
ROC-AUC: 0.8640 ± 0.0042
------------------------------
Mean scores with standard deviation for KNeighbours:
Accuracy: 0.8198 ± 0.0018
Precision: 0.6423 ± 0.0137
Recall: 0.5568 ± 0.0085
F1 Score: 0.5964 ± 0.0083
ROC-AUC: 0.8556 ± 0.0040
------------------------------
Mean scores with standard deviation for LogisticRegression:
Accuracy: 0.8268 ± 0.0021
Precision: 0.6799 ± 0.0171
Recall: 0.5213 ± 0.0069
F1 Score: 0.5900 ± 0.0100
ROC-AUC: 0.8779 ± 0.0024


# GridSearch with Custom StratKfold
- GS: Custom
- StratKFold: Custom
- CV: Custom

Best parameters found:  {'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 1}
Best parameters found:  {'n_neighbors': 9, 'weights': 'uniform', 'metric': 'manhattan'}
Best parameters found:  {'C': 15, 'penalty': 'l1', 'solver': 'liblinear'}
Mean scores with standard deviation For DecisionTreeClassifier:
Accuracy: 0.8252 ± 0.0023
Precision: 0.7297 ± 0.0085
Recall: 0.4283 ± 0.0122
F1 Score: 0.5396 ± 0.0096
ROC-AUC: 0.8726 ± 0.0027
------------------------------
Mean scores with standard deviation for KNeighbours:
Accuracy: 0.8215 ± 0.0040
Precision: 0.6513 ± 0.0108
Recall: 0.5469 ± 0.0099
F1 Score: 0.5945 ± 0.0092
ROC-AUC: 0.8531 ± 0.0031
------------------------------
Mean scores with standard deviation for LogisticRegression:
Accuracy: 0.8261 ± 0.0020
Precision: 0.6781 ± 0.0049
Recall: 0.5202 ± 0.0109
F1 Score: 0.5886 ± 0.0073
ROC-AUC: 0.8778 ± 0.0033