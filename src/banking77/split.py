from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def make_stratified_split_ids(train_df: pd.DataFrame, test_size=0.2, random_state=42):
    dev, val = train_test_split(
        train_df,
        test_size=test_size,
        random_state=random_state,
        stratify=train_df["intent"]
    )
    return dev[["row_id"]].copy(), val[["row_id"]].copy()


def save_split_ids(dev_ids: pd.DataFrame, val_ids: pd.DataFrame, out_dir="artifacts/splits"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev_fp = out / "day17_dev_ids.csv"
    val_fp = out / "day17_val_ids.csv"
    dev_ids.to_csv(dev_fp, index=False)
    val_ids.to_csv(val_fp, index=False)
    return dev_fp, val_fp


def load_split_ids(split_dir="artifacts/splits"):
    split_dir = Path(split_dir)
    dev = pd.read_csv(split_dir / "day17_dev_ids.csv")
    val = pd.read_csv(split_dir / "day17_val_ids.csv")
    return dev, val


def apply_frozen_split(train_df: pd.DataFrame, dev_ids: pd.DataFrame, val_ids: pd.DataFrame):
    dev = train_df.merge(dev_ids, on="row_id", how="inner")
    val = train_df.merge(val_ids, on="row_id", how="inner")
    return dev, val


def verify_validation_alignment(val_df: pd.DataFrame, baseline_preds: pd.DataFrame):
    """
    Ensure exact row_id and true label alignment with Day 17 baseline predictions.
    """
    req = {"row_id", "true_intent"}
    if not req.issubset(baseline_preds.columns):
        raise ValueError(f"Baseline predictions missing required columns: {req - set(baseline_preds.columns)}")

    left = val_df[["row_id", "intent"]].sort_values("row_id").reset_index(drop=True)
    right = baseline_preds[["row_id", "true_intent"]].sort_values("row_id").reset_index(drop=True)

    if len(left) != len(right):
        raise RuntimeError(f"Validation size mismatch: {len(left)} vs {len(right)}")
    if not left["row_id"].equals(right["row_id"]):
        raise RuntimeError("Validation row_id mismatch vs Day 17 baseline.")
    if not left["intent"].equals(right["true_intent"]):
        raise RuntimeError("Validation true label mismatch vs Day 17 baseline.")