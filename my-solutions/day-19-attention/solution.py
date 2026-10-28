"""
Day 20 — Attention model vs Day 19 GRU baseline on frozen Day 17 split.

Outputs (under data/banking77/artifacts):
- results/day19_attention_metrics.csv
- results/day19_val_predictions.csv
- results/day19_error_migration_vs_day19.csv
- results/day19_targeted_confusions.csv
- results/day19_attention_examples.csv
- results/day19_confusion_pairs.csv
- results/day19_training_history.csv
- models/day19_attention_model.pt
"""

import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
from src.banking77.io import load_train_test, to_canonical, basic_quality_filter
from src.banking77.split import load_split_ids, apply_frozen_split, verify_validation_alignment
from src.banking77.metrics import (
    classification_summary,
    directed_confusion_pairs,
    per_class_report_df,
    build_prediction_frame,
)
from src.banking77.analysis import error_migration, targeted_confusion_delta

# ----------------------------
# Config
# ----------------------------
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
ATTN_EXAMPLES = 10

ART = Path("data/banking77/artifacts")
RES = ART / "results"
MOD = ART / "models"
RES.mkdir(parents=True, exist_ok=True)
MOD.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Utilities
# ----------------------------
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
    ln = len(ids)
    if ln < max_len:
        ids += [0] * (max_len - ln)
    return ids, ln, toks[:max_len]


class TextDataset(Dataset):
    def __init__(self, df, stoi, label_to_idx, max_len=48):
        self.row_id = df["row_id"].tolist()
        self.text = df["text"].tolist()
        self.y = [label_to_idx[y] for y in df["intent"].tolist()]
        self.stoi = stoi
        self.max_len = max_len

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        ids, ln, toks = encode_text(self.text[i], self.stoi, self.max_len)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(ln, dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
            self.row_id[i],
            self.text[i],
            toks,
        )


def collate_fn(batch):
    x, ln, y, row_id, text, toks = zip(*batch)
    return (
        torch.stack(x),
        torch.stack(ln),
        torch.stack(y),
        list(row_id),
        list(text),
        list(toks),
    )


# ----------------------------
# Model: GRU + additive attention
# ----------------------------
class GRUAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_classes, pad_idx=0, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=False)
        self.drop = nn.Dropout(dropout)

        # additive attention
        self.attn_W = nn.Linear(hid_dim, hid_dim, bias=True)
        self.attn_v = nn.Linear(hid_dim, 1, bias=False)

        self.fc = nn.Linear(hid_dim, num_classes)

    def forward(self, x, lengths, return_attention=False):
        """
        x: [B, T]
        lengths: [B]
        """
        emb = self.emb(x)  # [B,T,E]
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        H, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1)
        )  # [B,T,H]

        # attention scores
        e = self.attn_v(torch.tanh(self.attn_W(H))).squeeze(-1)  # [B,T]

        # mask pads
        mask = x.eq(0)  # True for PAD
        e = e.masked_fill(mask, float("-inf"))

        a = torch.softmax(e, dim=1)  # [B,T], sums to 1 over valid tokens
        context = torch.bmm(a.unsqueeze(1), H).squeeze(1)  # [B,H]
        context = self.drop(context)
        logits = self.fc(context)

        if return_attention:
            return logits, a
        return logits


def evaluate(model, loader, device, with_attention=False):
    model.eval()
    ce = nn.CrossEntropyLoss()

    all_true, all_pred, all_score = [], [], []
    all_row, all_text, all_tokens = [], [], []
    all_attn = []
    total_loss = 0.0

    with torch.no_grad():
        for x, ln, y, row_id, text, toks in loader:
            x, ln, y = x.to(device), ln.to(device), y.to(device)

            if with_attention:
                logits, attn = model(x, ln, return_attention=True)
                all_attn.extend(attn.cpu().numpy().tolist())
            else:
                logits = model(x, ln, return_attention=False)

            loss = ce(logits, y)
            total_loss += loss.item() * len(y)

            probs = torch.softmax(logits, dim=1)
            score, pred = probs.max(dim=1)

            all_true.extend(y.cpu().numpy().tolist())
            all_pred.extend(pred.cpu().numpy().tolist())
            all_score.extend(score.cpu().numpy().tolist())
            all_row.extend(row_id)
            all_text.extend(text)
            all_tokens.extend(toks)

    avg_loss = total_loss / max(1, len(all_true))

    out = {
        "loss": avg_loss,
        "y_true_idx": all_true,
        "y_pred_idx": all_pred,
        "score": all_score,
        "row_id": all_row,
        "text": all_text,
        "tokens": all_tokens,
    }
    if with_attention:
        out["attn"] = all_attn
    return out


def build_attention_examples(df_pred, tokens_list, attn_list, n=10):
    """
    Select mixed examples:
      - highest-score wrong
      - highest-score correct
    Save token-weight pairs for quick inspection.
    """
    tmp = df_pred.copy()
    tmp["idx"] = np.arange(len(tmp))

    wrong = tmp[~tmp["correct"]].sort_values("model_score", ascending=False).head(n // 2)
    correct = tmp[tmp["correct"]].sort_values("model_score", ascending=False).head(n - len(wrong))
    pick = pd.concat([wrong, correct], axis=0).sort_values("idx")

    rows = []
    for r in pick.itertuples(index=False):
        i = int(r.idx)
        toks = tokens_list[i]
        w = attn_list[i][:len(toks)]  # trim to actual tokens (before padding)
        if len(toks) == 0:
            continue
        # normalize just in case of tiny numeric drift
        s = float(np.sum(w))
        if s > 0:
            w = [float(x / s) for x in w]
        rows.append({
            "row_id": r.row_id,
            "correct": bool(r.correct),
            "true_intent": r.true_intent,
            "predicted_intent": r.predicted_intent,
            "model_score": float(r.model_score),
            "tokens": toks,
            "attention_weights": w,
            "top_tokens": [toks[j] for j in np.argsort(w)[::-1][:5]] if len(toks) > 0 else [],
            "top_weights": sorted(w, reverse=True)[:5] if len(w) > 0 else [],
        })

    return pd.DataFrame(rows)


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load + canonical + minimal clean
    raw_train, _ = load_train_test()
    train = to_canonical(raw_train, "train")
    train = basic_quality_filter(train).reset_index(drop=True)

    # 2) Frozen split from Day 17
    dev_ids, val_ids = load_split_ids("data/banking77/artifacts/splits")
    dev, val = apply_frozen_split(train, dev_ids, val_ids)

    # 3) Validate alignment with Day 17 rows
    day17_preds = pd.read_csv("data/banking77/artifacts/results/day17_val_predictions.csv")
    verify_validation_alignment(val, day17_preds)

    # 4) Day 19 and Day 17 artifacts for comparison
    day19_preds = pd.read_csv("data/banking77/artifacts/results/day19_val_predictions.csv")
    day17_conf = pd.read_csv("data/banking77/artifacts/results/day17_confusion_pairs.csv")

    # 5) Label maps
    labels = sorted(dev["intent"].unique().tolist())
    label_to_idx = {y: i for i, y in enumerate(labels)}
    idx_to_label = {i: y for y, i in label_to_idx.items()}

    # 6) Vocab from dev only
    stoi = build_vocab(dev["text"].tolist(), min_freq=MIN_FREQ)
    vocab_size = len(stoi)

    # 7) Dataset/loaders
    dev_ds = TextDataset(dev, stoi, label_to_idx, max_len=MAX_LEN)
    val_ds = TextDataset(val, stoi, label_to_idx, max_len=MAX_LEN)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 8) Model
    model = GRUAttentionClassifier(
        vocab_size=vocab_size,
        emb_dim=EMB_DIM,
        hid_dim=HID_DIM,
        num_classes=len(labels),
        pad_idx=0,
        dropout=DROPOUT,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()

    # 9) Train (early stop on val macro F1)
    best_state = None
    best_f1 = -1.0
    wait = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        tr_loss = 0.0
        n = 0

        for x, ln, y, _, _, _ in dev_loader:
            x, ln, y = x.to(device), ln.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x, ln, return_attention=False)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

            tr_loss += loss.item() * len(y)
            n += len(y)

        tr_loss /= max(1, n)

        ev = evaluate(model, val_loader, device, with_attention=False)
        y_true = [idx_to_label[i] for i in ev["y_true_idx"]]
        y_pred = [idx_to_label[i] for i in ev["y_pred_idx"]]
        m = classification_summary(y_true, y_pred)
        m["epoch"] = epoch
        m["train_loss"] = tr_loss
        m["val_loss"] = ev["loss"]
        m["epoch_seconds"] = time.time() - t0
        history.append(m)

        print(f"Epoch {epoch} | train_loss={tr_loss:.4f} val_loss={ev['loss']:.4f} macro_f1={m['macro_f1']:.4f}")

        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping.")
                break

    # restore best
    model.load_state_dict(best_state)

    # 10) Final eval WITH attention
    ev = evaluate(model, val_loader, device, with_attention=True)
    y_true = [idx_to_label[i] for i in ev["y_true_idx"]]
    y_pred = [idx_to_label[i] for i in ev["y_pred_idx"]]

    pred_df = build_prediction_frame(
        row_ids=ev["row_id"],
        texts=ev["text"],
        y_true=y_true,
        y_pred=y_pred,
        y_score=ev["score"],
    )
    pred_df.to_csv(RES / "day19_val_predictions.csv", index=False)

    # 11) Metrics compare Day 19 vs Day 20
    m20 = classification_summary(pred_df["true_intent"], pred_df["predicted_intent"])
    m19 = classification_summary(day19_preds["true_intent"], day19_preds["predicted_intent"])

    metrics_df = pd.DataFrame([
        {"model": "day19_gru", **m19},
        {"model": "day19_gru_attention", **m20},
    ])
    metrics_df["delta_vs_day19"] = metrics_df["macro_f1"] - m19["macro_f1"]
    metrics_df.to_csv(RES / "day19_attention_metrics.csv", index=False)

    # 12) confusion + targeted confusions
    conf20 = directed_confusion_pairs(
        pred_df["true_intent"], pred_df["predicted_intent"],
        labels=sorted(train["intent"].unique())
    )
    conf20.to_csv(RES / "day19_confusion_pairs.csv", index=False)

    targeted = targeted_confusion_delta(day17_conf, pred_df, top_k=10)
    targeted.to_csv(RES / "day19_targeted_confusions.csv", index=False)

    # 13) Migration vs Day 19
    mig = error_migration(day19_preds, pred_df)
    mig.to_csv(RES / "day19_error_migration_vs_day19.csv", index=False)

    # 14) attention examples
    attn_examples = build_attention_examples(
        df_pred=pred_df,
        tokens_list=ev["tokens"],
        attn_list=ev["attn"],
        n=ATTN_EXAMPLES
    )
    attn_examples.to_csv(RES / "day19_attention_examples.csv", index=False)

    # 15) Save model and run metadata
    pd.DataFrame(history).to_csv(RES / "day19_training_history.csv", index=False)
    per_class_report_df(
        pred_df["true_intent"], pred_df["predicted_intent"],
        labels=labels, target_names=[str(x) for x in labels]
    ).to_csv(RES / "day19_per_class_report.csv")

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
                "architecture": "gru_additive_attention",
            },
        },
        MOD / "day19_attention_model.pt",
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
                "note": "Attention weights are diagnostic, not causal attribution.",
            },
            f,
            indent=2,
        )

    print("\nSaved Day 20 outputs in data/banking77/artifacts/results and models.")
    print(metrics_df.to_string(index=False))
    print("\nMigration summary:")
    print(mig["status"].value_counts().to_string())
    print("\nTop targeted confusion deltas:")
    print(targeted.head(10).to_string(index=False))


if __name__ == "__main__":
    main()