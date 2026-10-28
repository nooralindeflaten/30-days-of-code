import json
from pathlib import Path
import numpy as np, pandas as pd
import os, sys
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
from src.taco.evaluation import iou_matrix, match

# ---- config (edit here) ----
ANN = Path("data/TACO/annotations.json")
SPLIT, MAP = Path("data/TACO/processed/split_manifest.csv"), Path("data/TACO/processed/category_mapping.json")
PRED = Path("data/TACO/reports/day25/validation_predictions.json")
OUT = Path("data/TACO/reports/day26"); OUT.mkdir(parents=True, exist_ok=True)
IOU, REF_THR = 0.5, 0.5
THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# ---- ground truth (mapped to model categories) and predictions, per val image ----
coco = json.loads(ANN.read_text())
orig2model = {int(k): v["model_category_id"] for k, v in json.loads(MAP.read_text())["original_to_model"].items()}
val_ids = {int(i) for i in pd.read_csv(SPLIT).query("split == 'val'")["image_id"]}
gt = {i: ([], []) for i in val_ids}
for a in coco["annotations"]:
    if a["image_id"] in val_ids and a["category_id"] in orig2model:
        x, y, w, h = a["bbox"]
        gt[a["image_id"]][0].append([x, y, x + w, y + h]); gt[a["image_id"]][1].append(orig2model[a["category_id"]])
pr = {i: ([], [], []) for i in val_ids}
for p in json.loads(PRED.read_text()):
    pr[p["image_id"]][0].append(p["bbox_xyxy"]); pr[p["image_id"]][1].append(p["category_id"]); pr[p["image_id"]][2].append(p["score"])

size = lambda b: "small" if (b[2] - b[0]) * (b[3] - b[1]) < 32 ** 2 else "medium" if (b[2] - b[0]) * (b[3] - b[1]) < 96 ** 2 else "large"


def evaluate(thr):
    """Returns per-image rows of error events at one score threshold."""
    rows = []
    for iid in sorted(val_ids):
        gb, gl = np.array(gt[iid][0]).reshape(-1, 4), np.array(gt[iid][1])
        pb, pl, ps = np.array(pr[iid][0]).reshape(-1, 4), np.array(pr[iid][1]), np.array(pr[iid][2])
        k = ps >= thr; pb, pl, ps = pb[k], pl[k], ps[k]
        order, status, fn = match(gb, gl, pb, pl, ps, IOU)
        best_iou = iou_matrix(pb[order], gb).max(axis=1) if len(pb) and len(gb) else np.zeros(len(pb))
        for s, bi, i in zip(status, best_iou, order):
            kind = "tp" if s == "tp" else ("fp_duplicate_or_wrong_class" if bi >= IOU else "fp_background")
            rows.append({"image_id": iid, "kind": kind, "score": float(ps[i]), "size": size(pb[i])})
        for j in np.where(fn)[0]:
            rows.append({"image_id": iid, "kind": "missed", "score": None, "size": size(gb[j])})
        if len(pb) == 0:
            rows.append({"image_id": iid, "kind": "image_no_detections", "score": None, "size": None})
    return pd.DataFrame(rows)


# 1) error taxonomy at the reference threshold (by kind and object size)
tax = evaluate(REF_THR)
tax.groupby(["kind", "size"], dropna=False).size().rename("count").reset_index().to_csv(OUT / "error_taxonomy.csv", index=False)

# 2) threshold sweep (denominator: number of validation images)
sweep = []
for t in THRESHOLDS:
    e = evaluate(t)
    tp, fp, fn = (e.kind == "tp").sum(), e.kind.str.startswith("fp").sum(), (e.kind == "missed").sum()
    sweep.append({"threshold": t, "tp": tp, "fp": fp, "missed": fn,
                  "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                  "false_alarms_per_image": fp / len(val_ids), "misses_per_image": fn / len(val_ids),
                  "images_no_detections": (e.kind == "image_no_detections").sum()})
sweep = pd.DataFrame(sweep); sweep.to_csv(OUT / "threshold_table.csv", index=False)
print(sweep.round(3).to_string(index=False))

# 3) review policy note, filled with measured numbers (no invented costs)
r = sweep[sweep.threshold == REF_THR].iloc[0]
(OUT / "review_policy.md").write_text(f"""# Day 26 review policy (validation-only prototype)

At score threshold {REF_THR} and IoU {IOU} on {len(val_ids)} validation images:
precision {r.precision:.3f}, recall {r.recall:.3f}, {r.false_alarms_per_image:.2f} false alarms/image,
{r.misses_per_image:.2f} missed objects/image, {int(r.images_no_detections)} images with no detections.

Workflow: show every detection above the threshold (box + score) to a human reviewer; flag duplicates and
low-score detections for extra scrutiny; spot-check no-detection images.

Cautions: scores are not calibrated probabilities. A missed object has no box to review, and an image with
no detections is not verified litter-free. These are development estimates on TACO validation images only.
""")