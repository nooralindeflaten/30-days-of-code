from pathlib import Path
import pandas as pd
import kagglehub


def normalize_text_key(s: pd.Series) -> pd.Series:
    """Diagnostic normalization only (not model preprocessing)."""
    return (
        s.astype("string")
         .str.strip()
         .str.replace(r"\s+", " ", regex=True)
         .str.casefold()
    )


def download_banking77(output_dir="data/banking77") -> Path:
    root = Path(kagglehub.dataset_download("sssonnn/banking77", output_dir=output_dir))
    return root


def _locate_single_csv(root: Path, filename: str) -> Path:
    direct = root / filename
    matches = [direct] if direct.is_file() else sorted(root.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {filename} under {root}, found: {matches}")
    return matches[0]


def load_train_test(output_dir="data/banking77"):
    root = download_banking77(output_dir)
    train_fp = _locate_single_csv(root, "train.csv")
    test_fp = _locate_single_csv(root, "test.csv")
    train = pd.read_csv(train_fp)
    test = pd.read_csv(test_fp)
    return train, test


def resolve_schema(df: pd.DataFrame):
    """
    Return detected text + numeric label + optional label_text columns.
    Expects BANKING77-like schema but does not assume blindly.
    """
    cols = {c.lower().strip(): c for c in df.columns}
    text_col = cols.get("text") or cols.get("query") or cols.get("utterance") or cols.get("sentence")
    label_col = cols.get("label") or cols.get("intent") or cols.get("category")
    label_text_col = cols.get("label_text")

    if text_col is None or label_col is None:
        raise ValueError(f"Could not resolve required schema from columns: {list(df.columns)}")
    return text_col, label_col, label_text_col


def to_canonical(frame: pd.DataFrame, split_name: str):
    """
    Canonical columns:
      row_id, text, intent, [label_text]
    """
    text_col, label_col, label_text_col = resolve_schema(frame)
    out = frame.rename(columns={text_col: "text", label_col: "intent"}).copy()
    out.insert(0, "row_id", [f"{split_name}_{i}" for i in range(len(out))])

    keep = ["row_id", "text", "intent"]
    if label_text_col is not None and label_text_col in out.columns:
        keep.append(label_text_col)
        out = out.rename(columns={label_text_col: "label_text"})

    out = out[keep].copy()
    return out


def basic_quality_filter(train: pd.DataFrame):
    """
    Keep rows with non-empty text and non-null intent.
    Remove conflicting normalized texts (same text key, multiple intents).
    """
    key = normalize_text_key(train["text"])
    valid = key.notna() & key.ne("") & train["intent"].notna()
    clean = train.loc[valid].copy()

    conflict_counts = pd.DataFrame({"key": key.loc[valid], "intent": clean["intent"]}).groupby("key")["intent"].nunique()
    conflict_keys = set(conflict_counts[conflict_counts > 1].index)
    if conflict_keys:
        clean_key = normalize_text_key(clean["text"])
        clean = clean.loc[~clean_key.isin(conflict_keys)].copy()

    return clean