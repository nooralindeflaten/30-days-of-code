import json, random
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
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
from src.taco.data import TacoDetectionDataset

# ---- config (edit here) ----
DATA = Path("data/TACO")
ANN = DATA / "annotations.json"          # must be the same file used on Day 22
SPLIT, MAP = Path("data/TACO/processed/split_manifest.csv"), Path("data/TACO/processed/category_mapping.json")
OUT = Path("data/TACO/reports/day25"); OUT.mkdir(parents=True, exist_ok=True)
Path("data/TACO/models").mkdir(exist_ok=True); Path("data/TACO/configs").mkdir(exist_ok=True)
EPOCHS, BATCH, LR, SEED = 5, 2, 5e-4, 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mapping = json.loads(MAP.read_text())
orig2model = {int(k): v["model_category_id"] for k, v in mapping["original_to_model"].items()}
num_classes = max(c["model_category_id"] for c in mapping["model_categories"]) + 1  # + background

mk = lambda name, train: TacoDetectionDataset(DATA, ANN, SPLIT, MAP, name, (1024, 1024), train, SEED)
collate = lambda b: tuple(zip(*b))
train_dl = DataLoader(mk("train", True), BATCH, shuffle=True, collate_fn=collate)
val_ds = mk("val", False)
val_dl = DataLoader(val_ds, BATCH, shuffle=False, collate_fn=collate)

model = fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, num_classes)
model.to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)


def run_loss(loader, train):
    """Mean detection loss. Validation loss is computed in train mode under no_grad (torchvision needs targets for loss)."""
    model.train()
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for imgs, tgts in loader:
            imgs = [i.to(dev) for i in imgs]
            tgts = [{"boxes": t["boxes"].to(dev), "labels": t["labels"].to(dev)} for t in tgts]
            loss = sum(model(imgs, tgts).values())
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); n += 1
    return total / max(n, 1)


# ---- train; choose checkpoint on VALIDATION loss ----
best, hist = float("inf"), []
for ep in range(1, EPOCHS + 1):
    tr, va = run_loss(train_dl, True), run_loss(val_dl, False)
    hist.append({"epoch": ep, "train_loss": tr, "val_loss": va}); print(hist[-1])
    if va < best:
        best = va; torch.save({"state_dict": model.state_dict()}, "data/TACO/models/detector.pt")
pd.DataFrame(hist).to_csv(OUT / "train_history.csv", index=False)
model.load_state_dict(torch.load("data/TACO/models/detector.pt")["state_dict"])

# ---- predict on val, mapped BACK to original image coordinates ----
model.eval(); preds = []
with torch.no_grad():
    for imgs, tgts in val_dl:
        for o, t in zip(model([i.to(dev) for i in imgs]), tgts):
            m = t["transform_meta"]
            b = o["boxes"].cpu().numpy().copy()
            b[:, [0, 2]] = (b[:, [0, 2]] - m["pad_x"]) / m["scale"]
            b[:, [1, 3]] = (b[:, [1, 3]] - m["pad_y"]) / m["scale"]
            for box, s, l in zip(b, o["scores"].cpu().numpy(), o["labels"].cpu().numpy()):
                if s >= 0.05:
                    preds.append({"image_id": int(t["image_id"]), "category_id": int(l),
                                  "score": float(s), "bbox_xyxy": [float(v) for v in box]})
Path(OUT / "validation_predictions.json").write_text(json.dumps(preds))

# ---- COCO-style AP (pycocotools) on val images, ground truth in model categories ----
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
coco = json.loads(ANN.read_text())
val_ids = {int(i) for i in pd.read_csv(SPLIT).query("split == 'val'")["image_id"]}
gt = {"images": [i for i in coco["images"] if i["id"] in val_ids],
      "categories": [{"id": c["model_category_id"], "name": c["model_category_name"]} for c in mapping["model_categories"]],
      "annotations": [{"id": a["id"], "image_id": a["image_id"], "category_id": orig2model[a["category_id"]],
                       "bbox": a["bbox"], "area": a["bbox"][2] * a["bbox"][3], "iscrowd": 0}
                      for a in coco["annotations"] if a["image_id"] in val_ids and a["category_id"] in orig2model]}
cg = COCO(); cg.dataset = gt; cg.createIndex()
dets = [{"image_id": p["image_id"], "category_id": p["category_id"], "score": p["score"],
         "bbox": [p["bbox_xyxy"][0], p["bbox_xyxy"][1], p["bbox_xyxy"][2] - p["bbox_xyxy"][0], p["bbox_xyxy"][3] - p["bbox_xyxy"][1]]}
        for p in preds]
ev = COCOeval(cg, cg.loadRes(dets), "bbox"); ev.evaluate(); ev.accumulate(); ev.summarize()
names = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]
pd.DataFrame({"metric": names, "value": ev.stats[:6]}).to_csv(OUT / "metrics.csv", index=False)

# ---- 8 overlays (original coordinates, score >= 0.3) ----
(OUT / "overlays").mkdir(exist_ok=True)
files = {i["id"]: i["file_name"] for i in gt["images"]}
for iid in sorted({p["image_id"] for p in preds if p["score"] >= 0.3})[:8]:
    img = ImageOps.exif_transpose(Image.open(DATA / files[iid])).convert("RGB"); d = ImageDraw.Draw(img)
    for p in preds:
        if p["image_id"] == iid and p["score"] >= 0.3:
            d.rectangle(p["bbox_xyxy"], outline="lime", width=3)
            d.text((p["bbox_xyxy"][0], max(0, p["bbox_xyxy"][1] - 12)), f"{p['score']:.2f}", fill="lime")
    img.thumbnail((1200, 1200)); img.save(OUT / "overlays" / f"val_{iid}.jpg")

Path("data/TACO/configs/model.json").write_text(json.dumps(
    {"arch": "fasterrcnn_mobilenet_v3_large_320_fpn", "weights": "torchvision DEFAULT", "epochs": EPOCHS, "batch": BATCH,
     "lr": LR, "seed": SEED, "num_classes": num_classes, "best_val_loss": best, "selection": "validation loss"}, indent=2))
print(pd.read_csv(OUT / "metrics.csv"))