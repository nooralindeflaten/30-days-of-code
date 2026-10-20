import sys
import os

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to 
# the sys.path.
sys.path.append(parent)

# adding the grandparent directory to
# the sys.path.
grandparent = os.path.dirname(parent)
sys.path.append(grandparent)

from datasets import load_dataset
import pandas as pd
import os





path = "data/banking77"
dataset = load_dataset(
    "csv",
    data_files={
        "train": path + "/train.csv",
        "test": path + "/test.csv"
    },
    delimiter=","
)

train = dataset["train"].to_pandas()
test = dataset["test"].to_pandas()



print("\nTRAINING SHAPE:")
print(train.shape)

print("\nTEST SHAPE:")
print(test.shape)

print("\nTRAINING COLUMNS:")
print(train.columns.tolist())

print("\nTEST COLUMNS:")
print(test.columns.tolist())

print("\nFIRST FIVE ROWS:")
print(train.head().to_string(index=False))

print("\nCOLUMN TYPES:")
print(train.dtypes)



labels = set(train['label_text'].unique())
test_labels = set(test['label_text'].unique())

missing_labels = labels - test_labels
missing_labels_test = test_labels - labels
print("\nMISSING LABELS IN TEST SET:")
print(missing_labels)
print("\nMISSING LABELS IN TRAIN SET:")
print(missing_labels_test)



# Check for missing values in the training and test sets
columns_with_missing_values_train = train.columns[train.isnull().any()].tolist()
columns_with_missing_values_test = test.columns[test.isnull().any()].tolist()

def check_missing_values(train, test):
    """Check nulls, blank/whitespace-only texts, duplicate rows, 
    duplicate texts with conflicting labels, and 
    train/test overlap. Quantify issues before choosing whether to 
    remove anything. Document decisions and avoid using test examples 
    to design preprocessing."""
    
    # Check for null values
    print("\nCOLUMNS WITH NULL VALUES IN TRAIN SET:")
    print(columns_with_missing_values_train)
    print("\nCOLUMNS WITH NULL VALUES IN TEST SET:")
    print(columns_with_missing_values_test)

    # Check for blank or whitespace-only texts
    blank_texts_train = train[train['text'].str.strip() == '']
    blank_texts_test = test[test['text'].str.strip() == '']
    print("\nNUMBER OF BLANK OR WHITESPACE-ONLY TEXTS IN TRAIN SET:", len(blank_texts_train))
    print("NUMBER OF BLANK OR WHITESPACE-ONLY TEXTS IN TEST SET:", len(blank_texts_test))

    # Check for duplicate rows
    duplicate_rows_train = train[train.duplicated()]
    duplicate_rows_test = test[test.duplicated()]
    print("\nNUMBER OF DUPLICATE ROWS IN TRAIN SET:", len(duplicate_rows_train))
    print("NUMBER OF DUPLICATE ROWS IN TEST SET:", len(duplicate_rows_test))

    # Check for duplicate texts with conflicting labels
    conflicting_labels_train = train.groupby('text')['intent'].nunique()
    conflicting_labels_test = test.groupby('text')['intent'].nunique()
    conflicting_labels_train_count = (conflicting_labels_train > 1).sum()
    conflicting_labels_test_count = (conflicting_labels_test > 1).sum()
    print("\nNUMBER OF DUPLICATE TEXTS WITH CONFLICTING LABELS IN TRAIN SET:", conflicting_labels_train_count)
    print("NUMBER OF DUPLICATE TEXTS WITH CONFLICTING LABELS IN TEST SET:", conflicting_labels_test_count)

    # Check for train/test overlap
    overlap_texts = set(train['text']).intersection(set(test['text']))
    print("\nNUMBER OF OVERLAPPING TEXTS BETWEEN TRAIN AND TEST SETS:", len(overlap_texts))



"""
Day 17 (lean): data contract + audit -> frozen split -> Dummy vs TF-IDF+LogReg baseline.

Run from repo root: python -m workflows.day17_baseline
Console output: a handful of lines (shapes, audit counts, metrics, top confusions).
Official test set: used ONLY for audit counts (overlap). No test predictions.
"""
from pathlib import Path
from time import perf_counter
import json

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


from src.banking77.io import load_train_test, to_canonical, normalize_text_key, basic_quality_filter
from src.banking77.metrics import (
    classification_summary, per_class_report_df, directed_confusion_pairs, build_prediction_frame
)

SEED = 42
VAL_SIZE = 0.20
RES, SPL, MOD = Path("data/banking77/artifacts/results"), Path("data/banking77/artifacts/splits"), Path("data/banking77/artifacts/models")
for d in (RES, SPL, MOD):
    d.mkdir(parents=True, exist_ok=True)

# Fixed, deliberately simple preprocessing policy (no tuning on Day 17).
TFIDF = dict(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def audit(train, test):
    """Quantify issues only; the only cleaning decisions are the ones in basic_quality_filter."""
    tk, sk = normalize_text_key(train["text"]), normalize_text_key(test["text"])

    def stats(df, key):
        conflicts = (pd.DataFrame({"k": key, "y": df["intent"]})
                     .groupby("k")["y"].nunique().gt(1).sum())
        return {
            "rows": len(df),
            "n_intents": df["intent"].nunique(),
            "null_or_blank_text": int(key.isna().sum() + key.eq("").sum()),
            "duplicate_texts": int(key.duplicated().sum()),
            "texts_with_conflicting_labels": int(conflicts),
        }

    table = pd.DataFrame({"train": stats(train, tk), "test": stats(test, sk)})
    table.loc["train_test_text_overlap", "test"] = len(set(tk.dropna()) & set(sk.dropna()))
    table.to_csv(RES / "day17_data_audit.csv")
    print(table)


def main():
    # 1) Load + data contract
    raw_train, raw_test = load_train_test()
    train = to_canonical(raw_train, "train")
    test = to_canonical(raw_test, "test")
    print("train/test:", train.shape, test.shape)

    # Label mapping from the CSV itself (only if supplied; never invented)
    if "label_text" in train.columns:
        (train[["intent", "label_text"]].drop_duplicates().sort_values("intent")
         .to_csv(RES / "day17_label_mapping.csv", index=False))

    # 2) Audit (counts only) then minimal cleaning
    audit(train, test)
    clean = basic_quality_filter(train).reset_index(drop=True)
    print(f"kept {len(clean)}/{len(train)} train rows after minimal cleaning")

    # 3) Frozen split, saved by row_id
    dev, val = train_test_split(clean, test_size=VAL_SIZE, random_state=SEED, stratify=clean["intent"])
    dev[["row_id"]].to_csv(SPL / "day17_dev_ids.csv", index=False)
    val[["row_id"]].to_csv(SPL / "day17_val_ids.csv", index=False)

    # 4) Baselines (vectorizer fit on dev only, inside the pipeline)
    models = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "tfidf_logreg": Pipeline([
            ("tfidf", TfidfVectorizer(**TFIDF)),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]),
    }
    rows, preds = [], {}
    for name, model in models.items():
        t0 = perf_counter()
        model.fit(dev["text"], dev["intent"])
        fit_s = perf_counter() - t0
        pred = model.predict(val["text"])
        score = model.predict_proba(val["text"]).max(axis=1)  # model score, NOT calibrated confidence
        rows.append({"model": name, **classification_summary(val["intent"], pred), "fit_seconds": fit_s})
        preds[name] = build_prediction_frame(val["row_id"], val["text"], val["intent"], pred, score)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RES / "day17_baseline_metrics.csv", index=False)
    print(metrics.to_string(index=False))

    # 5) Day 18 handoff artifacts (TF-IDF model only)
    best = preds["tfidf_logreg"]
    best.to_csv(RES / "day17_val_predictions.csv", index=False)

    labels = sorted(clean["intent"].unique())
    per_class_report_df(best["true_intent"], best["predicted_intent"], labels=labels) \
        .to_csv(RES / "day17_per_class_report.csv")

    conf = directed_confusion_pairs(best["true_intent"], best["predicted_intent"], labels=labels)
    conf.to_csv(RES / "day17_confusion_pairs.csv", index=False)
    print("top confusions:\n", conf.head(10).to_string(index=False))

    joblib.dump(models["tfidf_logreg"], MOD / "day17_tfidf_logreg.joblib")
    (RES / "day17_run_metadata.json").write_text(json.dumps({
        "seed": SEED, "val_size": VAL_SIZE, "tfidf_params": TFIDF,
        "n_dev": len(dev), "n_val": len(val),
        "note": "Test set used for audit counts only; no test predictions.",
    }, indent=2))


if __name__ == "__main__":
    main()