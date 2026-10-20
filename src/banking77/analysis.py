import numpy as np
import pandas as pd


def error_migration(day_prev: pd.DataFrame, day_new: pd.DataFrame):
    """
    Inputs must have: row_id, true_intent, predicted_intent, model_score, correct
    """
    a = day_prev[["row_id", "true_intent", "predicted_intent", "model_score", "correct"]].rename(
        columns={
            "predicted_intent": "pred_prev",
            "model_score": "score_prev",
            "correct": "correct_prev",
        }
    )
    b = day_new[["row_id", "predicted_intent", "model_score", "correct"]].rename(
        columns={
            "predicted_intent": "pred_new",
            "model_score": "score_new",
            "correct": "correct_new",
        }
    )
    j = a.merge(b, on="row_id", how="inner", validate="one_to_one")

    j["status"] = np.select(
        [
            (~j["correct_prev"]) & (j["correct_new"]),
            (j["correct_prev"]) & (~j["correct_new"]),
            (~j["correct_prev"]) & (~j["correct_new"]),
        ],
        ["fixed", "new_error", "both_wrong"],
        default="both_correct"
    )
    return j


def targeted_confusion_delta(prev_pairs: pd.DataFrame, new_pred_df: pd.DataFrame, top_k=10):
    """
    prev_pairs: columns true_intent,predicted_intent,count (e.g., Day 17 top confusions)
    new_pred_df: row-level prediction frame for new model
    """
    top = prev_pairs.head(top_k).copy()
    rows = []
    for r in top.itertuples(index=False):
        n_new = int(((new_pred_df["true_intent"] == r.true_intent) &
                     (new_pred_df["predicted_intent"] == r.predicted_intent)).sum())
        rows.append({
            "true_intent": r.true_intent,
            "predicted_intent": r.predicted_intent,
            "prev_count": int(r.count),
            "new_count": n_new,
            "delta_new_minus_prev": n_new - int(r.count)
        })
    return pd.DataFrame(rows)