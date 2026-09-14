from sklearn.datasets import fetch_openml
import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.pipeline as pipeline
import sklearn.preprocessing as preprocessing
import sklearn.impute as impute
import sklearn.compose as compose



adult = fetch_openml(name="adult", version=2, as_frame=True)
df = adult.frame

# FIX: cast to int, not bool -- bool dtype matched neither the numeric nor
# categorical branch of the ColumnTransformer below, so this feature was
# silently being dropped from the final processed output.
df['is_migrant'] = (df['native-country'] != 'United-States').astype(int)

# occupation has a lot of unknown values, and in my opinion this feature can
# make the model overfit, so I will drop it
df = df.drop(columns=['native-country', 'occupation', 'fnlwgt', 'capital-loss', 'capital-gain', 'education'])
df['target'] = df['class'].apply(lambda x: 0 if x == '<=50K' else 1)
dataset = df.drop(columns=['class'])

def impute_missing_values(col, strategy):
    # missing values in the dataset are labeled unknown or in numerical columns as NaN
    if strategy == 'mean':
        col_impute = col.dropna().mean()
        return col.fillna(col_impute)

    elif strategy == 'most_frequent':
        col_impute = col.mode()[0]
        return col.fillna(col_impute)

    elif strategy == 'constant':
        # FIX: pandas raises an error filling a Categorical-dtype column
        # with a value that isn't already a registered category. Add it
        # first if needed, so this branch doesn't break the moment it's
        # actually used on a category-dtype column.
        if isinstance(col.dtype, pd.CategoricalDtype) and 'Unknown' not in col.cat.categories:
            col = col.cat.add_categories(['Unknown'])
        return col.fillna('Unknown')  # I'll ensure only cat columns will invoke this


def standardize_column(column):
    n = len(column)
    mean = sum(column) / n
    variance = sum((x - mean) ** 2 for x in column) / n
    std = math.sqrt(variance)
    standardized_column = [(x - mean) / std for x in column]
    return standardized_column


class Imputer:
    def __init__(self, strategy, type='categorical'):
        self.strategy = strategy
        self.type = type
        self.columns_to_impute = None

    def fit(self, X):
        if self.type == 'categorical':
            self.columns_to_impute = X.select_dtypes(include=['category']).columns
        elif self.type == 'numerical':
            # FIX: was include=['int'] only, which silently skipped any
            # column pandas had upcast to float64 (e.g. due to a NaN
            # somewhere in it) -- those columns then went unimputed and
            # poisoned standardize_column's mean/variance with NaN.
            self.columns_to_impute = X.select_dtypes(include=['int', 'float']).columns
        return self

    def transform(self, X):
        for col in self.columns_to_impute:
            X[col] = impute_missing_values(X[col], self.strategy)
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class OneHotEnc:
    def __init__(self):
        self.categorical_columns = None

    def fit(self, X):
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        return self

    def transform(self, X):
        return pd.get_dummies(X, columns=self.categorical_columns, drop_first=True)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class StandardScale:
    def __init__(self):
        self.numerical_columns = None

    def fit(self, X):
        self.numerical_columns = X.select_dtypes(include='int').columns
        return self

    def transform(self, X):
        for col in self.numerical_columns:
            X[col] = standardize_column(X[col])
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X):
        for name, step in self.steps:
            step.fit(X)
            X = step.transform(X)
        return self

    def transform(self, X):
        for name, step in self.steps:
            X = step.transform(X)
        return X

    def fit_transform(self, X):
        for name, step in self.steps:
            step.fit(X)
            X = step.transform(X)
        return X


class ColumnTransformer:
    def __init__(self, transformers):
        self.transformers = transformers

    def fit(self, X):
        for name, transformer, columns in self.transformers:
            transformer.fit(X[columns])
        return self

    def transform(self, X):
        transformed_data = []
        for name, transformer, columns in self.transformers:
            transformed_data.append(transformer.transform(X[columns]))
        return pd.concat(transformed_data, axis=1)

    def fit_transform(self, X):
        for name, transformer, columns in self.transformers:
            transformer.fit(X[columns])
        transformed_data = []
        for name, transformer, columns in self.transformers:
            transformed_data.append(transformer.transform(X[columns]))
        return pd.concat(transformed_data, axis=1)


X = dataset.drop(columns=['target'])  # FIX: drop() rejects both columns= and axis=1 together in recent pandas
y = dataset['target']

# FIX: added stratify=y given the known class imbalance in this dataset.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# FIX: reset every index to a clean 0..n-1 range immediately after the
# split, on X *and* y together, in the same place. This is the actual fix
# for the target-column bug -- doing it here means every downstream frame
# (X_train_processed, y_train, etc.) shares consistent indices, so later
# concatenation/assignment aligns correctly instead of silently producing
# NaNs. Much safer than resetting only one side right before assignment.
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

num_transformer = Pipeline(steps=[
    ('imputer', Imputer(strategy='mean', type='numerical')),
    ('scaler', StandardScale())
])

cat_transformer = Pipeline(steps=[
    ('imputer', Imputer(strategy='most_frequent', type='categorical')),
    ('onehot', OneHotEnc())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, X.select_dtypes(include=['int', 'float']).columns),
    ('cat', cat_transformer, X.select_dtypes(include=['object', 'category']).columns)
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

train_df = pd.DataFrame(X_train_processed)
train_df["target"] = y_train  # indices now match -- no reset_index needed here anymore

test_df = pd.DataFrame(X_test_processed)
test_df["target"] = y_test

# Sanity check worth running once: confirms the fix actually worked.
assert train_df["target"].isna().sum() == 0, "target column still has NaNs -- index alignment issue remains"
print(f"train_df shape: {train_df.shape}, target NaNs: {train_df['target'].isna().sum()}")
print(f"test_df shape: {test_df.shape}, target NaNs: {test_df['target'].isna().sum()}")


train_df.to_csv("data/training_dataset.csv", index=False)
test_df.to_csv("data/testing_dataset.csv", index=False)
full_df = pd.concat([train_df, test_df], axis=0)
full_df.to_csv("data/full_dataset.csv", index=False)

# ---------------------------- Data Preprocessing using sklearn Pipeline and ColumnTransformer ----------------------------

categorical_features = X.select_dtypes(include=['object', 'category']).columns
numerical_features = X.select_dtypes(include=['int', 'float']).columns

cat_transformer_sklearn = pipeline.Pipeline(steps=[
    ('imputer', impute.SimpleImputer(strategy='most_frequent')),
    ('onehot', preprocessing.OneHotEncoder(drop='first'))
])

num_transformer_sklearn = pipeline.Pipeline(steps=[
    ('imputer', impute.SimpleImputer(strategy='mean')),
    ('scaler', preprocessing.StandardScaler())
])

preprosessor_sklearn = compose.ColumnTransformer(transformers=[
    ('num', num_transformer_sklearn, numerical_features),
    ('cat', cat_transformer_sklearn, categorical_features)
])

train_df_sklearn = pd.DataFrame(preprosessor_sklearn.fit_transform(X_train))
test_df_sklearn = pd.DataFrame(preprosessor_sklearn.transform(X_test))

train_df_sklearn["target"] = y_train.reset_index(drop=True)
test_df_sklearn["target"] = y_test.reset_index(drop=True)

train_df_sklearn.to_csv("data/training_dataset_sklearn.csv", index=False)
test_df_sklearn.to_csv("data/testing_dataset_sklearn.csv", index=False)