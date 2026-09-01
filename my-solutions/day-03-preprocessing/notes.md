
## Values

**Balance of target class**
class
<=50K    37155
>50K     11687
Name: count, dtype: int64

# Sources
Feature engineering
https://www.kaggle.com/code/keitazoumana/eda-feature-engineering-machine-learning#Adult-Income-:-Exploratory-Analysis-And-Precition

https://github.com/itdxer/adult-dataset-analysis/blob/master/Data%20analysis.ipynb

Sklearn structure:
Ohe: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

Pipeline:
https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html



# Code from school

The pipeline structure of the code is inspired by the structure of this code from when I was studying Applied Machine Learning at Noroff, but I also have other scripts from school where I got some inspo.




```python
"""
Risk dataset preprocessing.
Steps:
1. Impute string features with most frequent value
2. Encode categorical features numerically
3. Train/test split (90/10)
4. Impute numeric with mean
5. Scale [-1, 1]
6. Apply L1 normalization
7. Recombine X + y and save CSVs
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# 1. Load dataset
df = pd.read_csv("../data/risk.csv")

# Features + label
X = df.drop("y", axis=1)
y = df["y"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Identify categorical vs numeric
cat_features = X.select_dtypes(include="object").columns
num_features = X.select_dtypes(exclude="object").columns

# Pipelines
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", MinMaxScaler(feature_range=(-1, 1))),
    ("normalizer", Normalizer(norm="l1"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_features),
    ("cat", categorical_transformer, cat_features)
])

# Fit transform
train_processed = preprocessor.fit_transform(X_train)
test_processed = preprocessor.transform(X_test)

# Convert back to DataFrame
train_df = pd.DataFrame(train_processed)
train_df["y"] = y_train.reset_index(drop=True)

test_df = pd.DataFrame(test_processed)
test_df["y"] = y_test.reset_index(drop=True)

# Save
train_df.to_csv("../outputs/Risk_train.csv", index=False)
test_df.to_csv("../outputs/Risk_test.csv", index=False)
```
