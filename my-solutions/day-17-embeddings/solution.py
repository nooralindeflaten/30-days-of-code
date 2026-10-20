"""
Day 18 (lean): sentence embeddings vs frozen Day 17 baseline on EXACT same validation rows.

Run:
  python -m workflows.day18_embeddings

Requirements:
  pip install sentence-transformers scikit-learn pandas numpy kagglehub joblib

Inputs:
  artifacts/splits/day17_dev_ids.csv
  artifacts/splits/day17_val_ids.csv
  artifacts/results/day17_val_predictions.csv
  artifacts/results/day17_confusion_pairs.csv

Outputs:
  artifacts/results/day18_embedding_metrics.csv
  artifacts/results/day18_val_predictions.csv
  artifacts/results/day18_error_migration.csv
  artifacts/results/day18_targeted_confusions.csv
  artifacts/results/day18_runtime_summary.json
  artifacts/models/day18_embedding_logreg.joblib
"""
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

from pathlib import Path
from time import perf_counter
import json
import random

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from src.banking77.io import load_train_test, to_canonical, basic_quality_filter
from src.banking77.split import load_split_ids, apply_frozen_split, verify_validation_alignment
from src.banking77.metrics import classification_summary, build_prediction_frame, directed_confusion_pairs
from src.banking77.analysis import error_migration, targeted_confusion_delta

# ----------------------------
# Config
# ----------------------------
SEED = 42
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = None   # optional: pin a specific revision hash
BATCH_SIZE = 128

ART = Path("data/banking77/artifacts")
RES = ART / "results"
MOD = ART / "models"
RES.mkdir(parents=True, exist_ok=True)
MOD.mkdir(parents=True, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def main():
    set_seed(SEED)

    # ----------------------------
    # 1) Load canonical train and apply frozen Day 17 split
    # ----------------------------
    raw_train, _ = load_train_test()
    train = to_canonical(raw_train, "train")
    train = basic_quality_filter(train).reset_index(drop=True)

    dev_ids, val_ids = load_split_ids("data/banking77/artifacts/splits")
    dev, val = apply_frozen_split(train, dev_ids, val_ids)

    # ----------------------------
    # 2) Verify alignment with Day 17 baseline predictions
    # ----------------------------
    day17_pred_fp = RES / "day17_val_predictions.csv"
    if not day17_pred_fp.exists():
        raise FileNotFoundError(f"Missing required input: {day17_pred_fp}")
    day17_preds = pd.read_csv(day17_pred_fp)

    # Ensures exact same validation row_id + true labels
    verify_validation_alignment(val, day17_preds)

    # Reorder val to match Day 17 order exactly (for paired analysis clarity)
    val = (
        day17_preds[["row_id"]]
        .merge(val, on="row_id", how="left", validate="one_to_one")
    )

    # ----------------------------
    # 3) Encode dev/val texts
    # ----------------------------
    t0 = perf_counter()
    encoder = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION)
    model_load_s = perf_counter() - t0

    t0 = perf_counter()
    X_dev = encoder.encode(
        dev["text"].tolist(),
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    dev_encode_s = perf_counter() - t0

    t0 = perf_counter()
    X_val = encoder.encode(
        val["text"].tolist(),
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    val_encode_s = perf_counter() - t0

    # ----------------------------
    # 4) Train simple classifier on embeddings
    # ----------------------------
    y_dev = dev["intent"].values
    y_val = val["intent"].values

    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    t0 = perf_counter()
    clf.fit(X_dev, y_dev)
    fit_s = perf_counter() - t0

    t0 = perf_counter()
    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)
    pred_s = perf_counter() - t0

    y_score = y_proba.max(axis=1)  # model score, not calibrated confidence

    # ----------------------------
    # 5) Save predictions and metrics
    # ----------------------------
    pred_df = build_prediction_frame(
        row_ids=val["row_id"],
        texts=val["text"],
        y_true=y_val,
        y_pred=y_pred,
        y_score=y_score,
    )
    pred_df.to_csv(RES / "day18_val_predictions.csv", index=False)

    m18 = classification_summary(y_val, y_pred)
    m17 = classification_summary(day17_preds["true_intent"], day17_preds["predicted_intent"])

    metrics_df = pd.DataFrame([
        {"model": "day17_tfidf_logreg", **m17},
        {"model": "day18_embeddings_logreg", **m18},
    ])
    metrics_df.to_csv(RES / "day18_embedding_metrics.csv", index=False)

    # ----------------------------
    # 6) Paired migration and targeted confusions
    # ----------------------------
    mig = error_migration(day17_preds, pred_df)
    mig.to_csv(RES / "day18_error_migration.csv", index=False)

    conf18 = directed_confusion_pairs(
        pred_df["true_intent"], pred_df["predicted_intent"],
        labels=sorted(train["intent"].unique())
    )
    conf18.to_csv(RES / "day18_confusion_pairs.csv", index=False)

    prev_conf_fp = RES / "day17_confusion_pairs.csv"
    if not prev_conf_fp.exists():
        raise FileNotFoundError(f"Missing required input: {prev_conf_fp}")
    conf17 = pd.read_csv(prev_conf_fp)

    targeted = targeted_confusion_delta(conf17, pred_df, top_k=10)
    targeted.to_csv(RES / "day18_targeted_confusions.csv", index=False)

    # ----------------------------
    # 7) Save model + runtime summary
    # ----------------------------
    joblib.dump(clf, MOD / "day18_embedding_logreg.joblib")

    runtime = {
        "seed": SEED,
        "embedding_model": MODEL_NAME,
        "embedding_revision": MODEL_REVISION,
        "batch_size": BATCH_SIZE,
        "n_dev": int(len(dev)),
        "n_val": int(len(val)),
        "embedding_dim": int(X_dev.shape[1]),
        "model_load_seconds": model_load_s,
        "dev_encode_seconds": dev_encode_s,
        "val_encode_seconds": val_encode_s,
        "classifier_fit_seconds": fit_s,
        "classifier_predict_seconds": pred_s,
        "val_ms_per_message": 1000.0 * (val_encode_s + pred_s) / max(1, len(val)),
        "embedding_arrays_mb": float((X_dev.nbytes + X_val.nbytes) / 1e6),
    }
    with open(RES / "day18_runtime_summary.json", "w", encoding="utf-8") as f:
        json.dump(runtime, f, indent=2)

    # Light console summary
    print("\n=== Day 18 complete ===")
    print(metrics_df.to_string(index=False))
    print("\nError migration:")
    print(mig["status"].value_counts().to_string())
    print("\nTop targeted confusion deltas:")
    print(targeted.head(10).to_string(index=False))


if __name__ == "__main__":
    main()