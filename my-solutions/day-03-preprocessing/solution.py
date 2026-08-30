from sklearn.datasets import fetch_openml
import pandas as pd
import math

adult = fetch_openml(name="adult", version=2, as_frame=True)
dataset = adult.frame


def standardize_column(column):
    n = len(column)
    mean = sum(column) / n
    variance = sum((x - mean) ** 2 for x in column) / n
    std = math.sqrt(variance)
    
    standardized_column = [(x - mean) / std for x in column]

    return standardized_column

def preprocess(df):
    df['is_migrant'] = df['native-country'] != 'United-States'
    df = df.astype({col: 'category' for col in df.select_dtypes(include='object').columns})
    print(df.info())

    cat_cols = df.select_dtypes(include=['category','object']).columns
    cat_cols = [col for col in cat_cols if col != 'class']
    num_cols = df.select_dtypes(include='int').columns
    num_df = df[num_cols]

    # Add unknown category for missing values in categorical columns
    for col in cat_cols:
        df[col] = df[col].cat.add_categories('Unknown').fillna('Unknown')
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    standardize_data = num_df.apply(standardize_column, axis=0)
    
    df_final = pd.concat([standardize_data, df_encoded], axis=1)
    # turn target into 2 classes <=50K and >50K
    df_final['target'] = df_final['class'].apply(lambda x: 0 if x == '<=50K' else 1)
    df_final = df_final.drop(columns=['class'])
    return df_final

def split_data(df, test_size=0.2):
    # Shuffle the DataFrame
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    df_shuffled.to_csv('data/processed_data.csv', index=False)
    # Split into training and test data and save
    test_data = df_shuffled.iloc[:int(test_size * len(df_shuffled))]
    train_data = df_shuffled.iloc[int(test_size * len(df_shuffled)):]

    # Save to CSV files
    train_data.to_csv('data/train_data.csv', index=False)
    test_data.to_csv('data/test_data.csv', index=False)

def main():
    df = dataset.copy()
    target = df['class']
    df_preprocessed = preprocess(df)
    split_data(df_preprocessed, test_size=0.2)
    
    
if __name__ == "__main__":
    main()    