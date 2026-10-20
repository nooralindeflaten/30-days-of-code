"""
Day 19 — Minimal sequence model (GRU) on frozen Day 17 split.
Outputs:
data/banking77/
- artifacts/results/day19_seq_metrics.csv
- artifacts/results/day19_val_predictions.csv
- artifacts/results/day19_error_migration_vs_day18.csv
- artifacts/results/day19_length_slice_metrics.csv
- artifacts/results/day19_confusion_pairs.csv
- artifacts/models/day19_sequence_model.pt
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
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.banking77.io import load_train_test, to_canonical, basic_quality_filter
from src.banking77.split import load_split_ids, apply_frozen_split, verify_validation_alignment
from src.banking77.metrics import (
    classification_summary,
    per_class_report_df,
    directed_confusion_pairs,
    build_prediction_frame
)
from src.banking77.analysis import error_migration

SEED = 42
MAX_LEN = 48
BATCH_SIZE = 128
EMB_DIM = 128
HID_DIM = 128
DROPOUT = 0.2
LR = 1e-3
EPOCHS = 8
PATIENCE = 2
MIN_FREQ = 2

ART = Path("data/banking77/artifacts")
RES = ART / "results"
MOD = ART / "models"
RES.mkdir(parents=True, exist_ok=True)
MOD.mkdir(parents=True, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def simple_tokenize(text: str):
    return str(text).lower().strip().split()


def build_vocab(texts, min_freq=2):
    from collections import Counter
    c = Counter()
    for t in texts:
        c.update(simple_tokenize(t))
    stoi = {"<pad>": 0, "<unk>": 1}
    for w, n in c.items():
        if n >= min_freq:
            stoi[w] = len(stoi)
    return stoi


def encode_text(text, stoi, max_len=48):
    toks = simple_tokenize(text)
    ids = [stoi.get(t, 1) for t in toks][:max_len]
    length = len(ids)
    if length < max_len:
        ids += [0] * (max_len - length)
    return ids, length


class TextDataset(Dataset):
    def __init__(self, df, stoi, label_to_idx, max_len=48):
        self.row_id = df["row_id"].tolist()
        self.text = df["text"].tolist()
        self.y = [label_to_idx[y] for y in df["intent"].tolist()]
        self.max_len = max_len
        self.stoi = stoi

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        ids, ln = encode_text(self.text[i], self.stoi, self.max_len)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(ln, dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
            self.row_id[i],
            self.text[i],
        )


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_classes, dropout=0.2, pad_idx=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim, num_classes)

    def forward(self, x, lengths):
        e = self.emb(x)  # [B,T,E]
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)  # h: [1,B,H]
        h = self.drop(h[-1])
        logits = self.fc(h)
        return logits


def evaluate(model, loader, device):
    model.eval()
    all_true, all_pred, all_score, all_row, all_text = [], [], [], [], []
    total_loss = 0.0
    ce = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, ln, y, row_id, text in loader:
            x, ln, y = x.to(device), ln.to(device), y.to(device)
            logits = model(x, ln)
            loss = ce(logits, y)
            total_loss += loss.item() * len(y)

            probs = torch.softmax(logits, dim=1)
            score, pred = probs.max(dim=1)

            all_true.extend(y.cpu().numpy().tolist())
            all_pred.extend(pred.cpu().numpy().tolist())
            all_score.extend(score.cpu().numpy().tolist())
            all_row.extend(list(row_id))
            all_text.extend(list(text))

    avg_loss = total_loss / max(1, len(all_true))
    return avg_loss, all_true, all_pred, all_score, all_row, all_text


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load canonical train
    raw_train, _ = load_train_test()
    train = to_canonical(raw_train, "train")
    train = basic_quality_filter(train).reset_index(drop=True)

    # 2) Apply frozen Day 17 split
    dev_ids, val_ids = load_split_ids("data/banking77/artifacts/splits")
    dev, val = apply_frozen_split(train, dev_ids, val_ids)

    # 3) Verify alignment with Day 17 baseline predictions
    day17_preds = pd.read_csv("data/banking77/artifacts/results/day17_val_predictions.csv")
    verify_validation_alignment(val, day17_preds)

    # 4) Load Day 18 predictions for migration analysis
    day18_preds = pd.read_csv("data/banking77/artifacts/results/day18_val_predictions.csv")

    # 5) Label mapping from DEV only
    labels = sorted(dev["intent"].unique().tolist())
    label_to_idx = {y: i for i, y in enumerate(labels)}
    idx_to_label = {i: y for y, i in label_to_idx.items()}

    # 6) Vocab from DEV only
    stoi = build_vocab(dev["text"].tolist(), min_freq=MIN_FREQ)
    vocab_size = len(stoi)

    # 7) Datasets/loaders
    dev_ds = TextDataset(dev, stoi, label_to_idx, max_len=MAX_LEN)
    val_ds = TextDataset(val, stoi, label_to_idx, max_len=MAX_LEN)

    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 8) Model
    model = GRUClassifier(
        vocab_size=vocab_size,
        emb_dim=EMB_DIM,
        hid_dim=HID_DIM,
        num_classes=len(labels),
        dropout=DROPOUT,
        pad_idx=0
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()

    # 9) Train with early stopping on val macro F1
    best_state = None
    best_f1 = -1
    wait = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        n = 0
        for x, ln, y, _, _ in dev_loader:
            x, ln, y = x.to(device), ln.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x, ln)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

            train_loss += loss.item() * len(y)
            n += len(y)

        train_loss /= max(1, n)
        val_loss, y_true_idx, y_pred_idx, y_score, row_ids, texts = evaluate(model, val_loader, device)

        y_true = [idx_to_label[i] for i in y_true_idx]
        y_pred = [idx_to_label[i] for i in y_pred_idx]
        m = classification_summary(y_true, y_pred)
        m["epoch"] = epoch
        m["train_loss"] = train_loss
        m["val_loss"] = val_loss
        m["epoch_seconds"] = time.time() - t0
        history.append(m)

        print(f"Epoch {epoch} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} macro_f1={m['macro_f1']:.4f}")

        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping.")
                break

    # restore best
    model.load_state_dict(best_state)

    # final val eval
    val_loss, y_true_idx, y_pred_idx, y_score, row_ids, texts = evaluate(model, val_loader, device)
    y_true = [idx_to_label[i] for i in y_true_idx]
    y_pred = [idx_to_label[i] for i in y_pred_idx]

    metrics = classification_summary(y_true, y_pred)
    metrics["val_loss"] = val_loss
    metrics["vocab_size"] = vocab_size
    metrics["max_len"] = MAX_LEN
    metrics["model"] = "gru_minimal"

    # save prediction frame
    pred_df = build_prediction_frame(
        row_ids=row_ids, texts=texts, y_true=y_true, y_pred=y_pred, y_score=y_score
    )
    pred_df.to_csv(RES / "day19_val_predictions.csv", index=False)

    # save metrics
    pd.DataFrame([metrics]).to_csv(RES / "day19_seq_metrics.csv", index=False)

    # per-class + confusion
    per_cls = per_class_report_df(y_true, y_pred, labels=labels, target_names=[str(x) for x in labels])
    per_cls.to_csv(RES / "day19_per_class_report.csv")

    conf = directed_confusion_pairs(y_true, y_pred, labels=labels)
    conf.to_csv(RES / "day19_confusion_pairs.csv", index=False)

    # migration vs day18
    mig = error_migration(day18_preds, pred_df)
    mig.to_csv(RES / "day19_error_migration_vs_day18.csv", index=False)

    # length-slice metrics
    tmp = pred_df.copy()
    tmp["n_words"] = tmp["text"].astype(str).str.split().str.len()
    q1, q2 = tmp["n_words"].quantile([0.33, 0.66]).values

    def bucket(n):
        if n <= q1:
            return "short"
        elif n <= q2:
            return "medium"
        return "long"

    tmp["length_bucket"] = tmp["n_words"].apply(bucket)

    slice_rows = []
    for b, g in tmp.groupby("length_bucket"):
        s = classification_summary(g["true_intent"], g["predicted_intent"])
        s["length_bucket"] = b
        s["n_obs"] = len(g)
        slice_rows.append(s)

    pd.DataFrame(slice_rows).to_csv(RES / "day19_length_slice_metrics.csv", index=False)

    # training history + model
    pd.DataFrame(history).to_csv(RES / "day19_training_history.csv", index=False)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "vocab_size": vocab_size,
                "emb_dim": EMB_DIM,
                "hid_dim": HID_DIM,
                "num_classes": len(labels),
                "dropout": DROPOUT,
                "max_len": MAX_LEN,
                "label_to_idx": label_to_idx,
                "stoi": stoi,
            },
        },
        MOD / "day19_sequence_model.pt",
    )

    with open(RES / "day19_run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": SEED,
                "device": str(device),
                "epochs_max": EPOCHS,
                "patience": PATIENCE,
                "batch_size": BATCH_SIZE,
                "learning_rate": LR,
                "min_freq": MIN_FREQ,
            },
            f,
            indent=2,
        )

    print("Saved Day 19 outputs in artifacts/results and artifacts/models.")


if __name__ == "__main__":
    main()