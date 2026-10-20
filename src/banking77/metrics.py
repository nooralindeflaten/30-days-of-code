import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def classification_summary(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def per_class_report_df(y_true, y_pred, labels=None, target_names=None):
    rep = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    return pd.DataFrame(rep).T


def directed_confusion_pairs(y_true, y_pred, labels=None):
    if labels is None:
        labels = sorted(pd.Series(y_true).unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for i, t in enumerate(labels):
        for j, p in enumerate(labels):
            if i == j:
                continue
            c = int(cm[i, j])
            if c > 0:
                rows.append({"true_intent": t, "predicted_intent": p, "count": c})
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return pd.DataFrame(columns=["true_intent", "predicted_intent", "count"])
    return out.sort_values("count", ascending=False).reset_index(drop=True)


def build_prediction_frame(row_ids, texts, y_true, y_pred, y_score):
    return pd.DataFrame({
        "row_id": np.array(row_ids),
        "text": np.array(texts),
        "true_intent": np.array(y_true),
        "predicted_intent": np.array(y_pred),
        "model_score": np.array(y_score),
        "correct": np.array(y_true) == np.array(y_pred)
    })