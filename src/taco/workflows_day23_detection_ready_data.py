"""
Day 23 — Detection-ready data pipeline (lean).

Purpose:
- Consume frozen Day 22 artifacts:
    data/processed/split_manifest.csv
    data/processed/category_mapping.json
- Build reusable Dataset + geometry-safe transforms
- Run minimal geometry tests and save a small overlay gallery

Run:
  python -m src.taco.workflows_day23_detection_ready_data --data-dir ./data/TACO

Outputs:
  src/taco/data.py
  configs/preprocessing.json
  reports/day23/geometry_report.json
  reports/day23/geometry_tests.json
  reports/day23/geometry_gallery/*.jpg
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

import torch
from torch.utils.data import Dataset

# torchvision v2 keeps image/boxes aligned when used with tv_tensors
import torchvision
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from collections import defaultdict

SEED = 42


# ----------------------------
# Utility helpers
# ----------------------------
def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def coco_xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    """
    Convert [x, y, w, h] -> [x1, y1, x2, y2].
    boxes_xywh: (N, 4)
    """
    out = boxes_xywh.copy().astype(np.float32)
    out[:, 2] = out[:, 0] + out[:, 2]
    out[:, 3] = out[:, 1] + out[:, 3]
    return out


def xyxy_to_coco_xywh(boxes_xyxy: np.ndarray) -> np.ndarray:
    """
    Convert [x1, y1, x2, y2] -> [x, y, w, h].
    """
    out = boxes_xyxy.copy().astype(np.float32)
    out[:, 2] = out[:, 2] - out[:, 0]
    out[:, 3] = out[:, 3] - out[:, 1]
    return out


def clip_boxes_xyxy(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    out = boxes.copy()
    out[:, 0] = np.clip(out[:, 0], 0, w)
    out[:, 1] = np.clip(out[:, 1], 0, h)
    out[:, 2] = np.clip(out[:, 2], 0, w)
    out[:, 3] = np.clip(out[:, 3], 0, h)
    return out


def valid_box_mask_xyxy(boxes: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return (w > eps) & (h > eps)


def draw_overlay(
    image_pil: Image.Image,
    boxes_xyxy: np.ndarray,
    labels: List[int],
    label_name_map: Dict[int, str],
    color: str = "red",
) -> Image.Image:
    canvas = image_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for b, lab in zip(boxes_xyxy, labels):
        x1, y1, x2, y2 = [float(v) for v in b]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, int(canvas.width / 500)))
        txt = label_name_map.get(int(lab), str(lab))
        draw.text((x1, max(0, y1 - 12)), txt, fill=color)
    return canvas


def inverse_letterbox_boxes(
    boxes_xyxy: np.ndarray, scale: float, pad_x: float, pad_y: float
) -> np.ndarray:
    """
    Inverse map from resized+padded canvas back to original coordinates.
    """
    out = boxes_xyxy.copy().astype(np.float32)
    out[:, [0, 2]] = (out[:, [0, 2]] - pad_x) / scale
    out[:, [1, 3]] = (out[:, [1, 3]] - pad_y) / scale
    return out


# ----------------------------
# Letterbox transform (deterministic)
# ----------------------------
@dataclass
class LetterboxMeta:
    scale: float
    pad_x: float
    pad_y: float
    src_w: int
    src_h: int
    dst_w: int
    dst_h: int


class DeterministicLetterbox:
    """
    Resize with preserved aspect ratio + symmetric padding to fixed size.
    Returns image tensor and transformed boxes + metadata.
    """

    def __init__(self, out_w: int = 1024, out_h: int = 1024):
        self.out_w = int(out_w)
        self.out_h = int(out_h)

    def __call__(self, image: Image.Image, boxes_xyxy: np.ndarray):
        src_w, src_h = image.size
        scale = min(self.out_w / src_w, self.out_h / src_h)
        new_w = int(round(src_w * scale))
        new_h = int(round(src_h * scale))

        # Resize image
        resized = image.resize((new_w, new_h), resample=Image.BILINEAR)

        # Pad to target canvas
        pad_x = (self.out_w - new_w) / 2.0
        pad_y = (self.out_h - new_h) / 2.0

        left = int(np.floor(pad_x))
        top = int(np.floor(pad_y))
        right = self.out_w - new_w - left
        bottom = self.out_h - new_h - top

        padded = ImageOps.expand(resized, border=(left, top, right, bottom), fill=(114, 114, 114))

        # Transform boxes
        boxes = boxes_xyxy.copy().astype(np.float32)
        if len(boxes):
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + left
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + top

        meta = LetterboxMeta(
            scale=float(scale),
            pad_x=float(left),
            pad_y=float(top),
            src_w=int(src_w),
            src_h=int(src_h),
            dst_w=self.out_w,
            dst_h=self.out_h,
        )
        return padded, boxes, meta


# ----------------------------
# Dataset
# ----------------------------
class TacoDetectionDataset(Dataset):
    """
    Returns:
      image_tensor: FloatTensor [3,H,W] normalized to [0,1]
      target: dict with:
        - boxes (FloatTensor [N,4], xyxy)
        - labels (Int64Tensor [N])
        - image_id (Int64Tensor [1])
        - orig_size (Int64Tensor [2] = [H,W])
        - size (Int64Tensor [2] = [H,W] after transform)
        - transform_meta (Python dict for inverse mapping)
    """

    def __init__(
        self,
        data_dir: Path,
        annotations_json: Path,
        split_manifest_csv: Path,
        category_mapping_json: Path,
        split_name: str,
        out_size: Tuple[int, int] = (1024, 1024),
        train_mode: bool = False,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.ann_file = Path(annotations_json)
        self.split_name = str(split_name)
        self.out_w, self.out_h = int(out_size[0]), int(out_size[1])
        self.train_mode = bool(train_mode)
        self.seed = int(seed)

        coco = json.loads(self.ann_file.read_text(encoding="utf-8"))

        self.images_df = pd.DataFrame(coco["images"])
        self.ann_df = pd.DataFrame(coco["annotations"])
        self.cat_df = pd.DataFrame(coco["categories"])

        # Load split
        split_df = pd.read_csv(split_manifest_csv)
        split_df = split_df[split_df["split"] == self.split_name].copy()

        # Keep only images that belong to this split
        self.image_ids = split_df["image_id"].astype(int).tolist()
        self.images_df = self.images_df[self.images_df["id"].isin(self.image_ids)].copy()
        self.images_df = self.images_df.sort_values("id").reset_index(drop=True)

        # Load mapping
        mapping = json.loads(Path(category_mapping_json).read_text(encoding="utf-8"))
        self.mapping_mode = mapping["mode"]

        # map original category -> model category id
        self.orig_to_model = {
            int(k): int(v["model_category_id"])
            for k, v in mapping["original_to_model"].items()
        }

        # model id -> name
        self.model_id_to_name = {}
        for m in mapping["model_categories"]:
            self.model_id_to_name[int(m["model_category_id"])] = str(m["model_category_name"])

        # group annotations per image (filter unknown categories and malformed bboxes lazily in __getitem__)
        ann_by_image = defaultdict(list)
        for rec in coco["annotations"]:
            ann_by_image[int(rec["image_id"])].append(rec)
        self.ann_by_image = ann_by_image

        # deterministic geometric transform
        self.letterbox = DeterministicLetterbox(out_w=self.out_w, out_h=self.out_h)

        # train-only augmentation (safe + simple): horizontal flip + mild color jitter
        # Apply these on PIL image + boxes manually for full control.
        self.rng = random.Random(self.seed)
        self.jitter = T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.02)

    def __len__(self):
        return len(self.images_df)

    def _apply_train_aug(self, image: Image.Image, boxes_xyxy: np.ndarray):
        # horizontal flip with p=0.5
        if self.rng.random() < 0.5:
            w, h = image.size
            image = ImageOps.mirror(image)
            if len(boxes_xyxy):
                x1 = boxes_xyxy[:, 0].copy()
                x2 = boxes_xyxy[:, 2].copy()
                boxes_xyxy[:, 0] = w - x2
                boxes_xyxy[:, 2] = w - x1

        # color jitter (image only)
        image = self.jitter(image)
        return image, boxes_xyxy

    def __getitem__(self, idx: int):
        row = self.images_df.iloc[idx]
        image_id = int(row["id"])
        file_name = str(row["file_name"])
        img_path = (self.data_dir / file_name).resolve()

        # load image
        try:
            with Image.open(img_path) as im:
                image = ImageOps.exif_transpose(im).convert("RGB")
        except (OSError, UnidentifiedImageError):
            # fallback: blank image preserving declared size
            image = Image.new("RGB", (int(row["width"]), int(row["height"])), (114, 114, 114))

        orig_w, orig_h = image.size

        anns = self.ann_by_image.get(image_id, [])

        boxes_xywh = []
        labels = []
        for a in anns:
            cat = int(a["category_id"])
            if cat not in self.orig_to_model:
                # unknown/unmapped category excluded explicitly
                continue

            bbox = a.get("bbox", None)
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue

            x, y, w, h = [float(v) for v in bbox]
            if w <= 0 or h <= 0:
                continue

            boxes_xywh.append([x, y, w, h])
            labels.append(self.orig_to_model[cat])

        if len(boxes_xywh):
            boxes_xywh = np.array(boxes_xywh, dtype=np.float32)
            boxes_xyxy = coco_xywh_to_xyxy(boxes_xywh)
            boxes_xyxy = clip_boxes_xyxy(boxes_xyxy, orig_w, orig_h)
            keep = valid_box_mask_xyxy(boxes_xyxy)
            boxes_xyxy = boxes_xyxy[keep]
            labels = np.array(labels, dtype=np.int64)[keep].tolist()
        else:
            boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
            labels = []

        # train-only aug BEFORE deterministic letterbox
        if self.train_mode:
            image, boxes_xyxy = self._apply_train_aug(image, boxes_xyxy)

        # deterministic transform (train+val)
        image_lb, boxes_lb, meta = self.letterbox(image, boxes_xyxy)
        boxes_lb = clip_boxes_xyxy(boxes_lb, self.out_w, self.out_h)
        keep2 = valid_box_mask_xyxy(boxes_lb)
        boxes_lb = boxes_lb[keep2]
        labels = np.array(labels, dtype=np.int64)[keep2].tolist()

        # to tensor [0,1]
        image_t = T.ToImage()(image_lb)
        image_t = T.ToDtype(torch.float32, scale=True)(image_t)

        target = {
            "boxes": torch.as_tensor(boxes_lb, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.as_tensor([image_id], dtype=torch.int64),
            "orig_size": torch.as_tensor([orig_h, orig_w], dtype=torch.int64),
            "size": torch.as_tensor([self.out_h, self.out_w], dtype=torch.int64),
            "transform_meta": {
                "scale": meta.scale,
                "pad_x": meta.pad_x,
                "pad_y": meta.pad_y,
                "src_w": meta.src_w,
                "src_h": meta.src_h,
                "dst_w": meta.dst_w,
                "dst_h": meta.dst_h,
            },
            "file_name": file_name,
        }

        return image_t, target


# ----------------------------
# Tests (lean, practical)
# ----------------------------
def test_double_flip_identity():
    """
    Horizontal flip twice should recover original boxes exactly (within tolerance).
    """
    w = 640
    boxes = np.array([[10, 20, 100, 120], [300, 5, 500, 40]], dtype=np.float32)

    def flip_once(b):
        out = b.copy()
        x1 = out[:, 0].copy()
        x2 = out[:, 2].copy()
        out[:, 0] = w - x2
        out[:, 2] = w - x1
        return out

    b2 = flip_once(flip_once(boxes))
    ok = np.allclose(b2, boxes, atol=1e-5)
    return {"name": "double_flip_identity", "pass": bool(ok)}


def test_letterbox_inverse():
    """
    Resize+pad then inverse mapping should approximately recover original boxes.
    """
    img = Image.new("RGB", (800, 600), (128, 128, 128))
    boxes = np.array([[40, 50, 200, 300], [300, 100, 780, 590]], dtype=np.float32)

    lb = DeterministicLetterbox(out_w=1024, out_h=1024)
    _, boxes_lb, meta = lb(img, boxes)

    boxes_inv = inverse_letterbox_boxes(
        boxes_lb, scale=meta.scale, pad_x=meta.pad_x, pad_y=meta.pad_y
    )
    ok = np.allclose(boxes_inv, boxes, atol=1.0)  # 1px tolerance due to rounding/padding
    return {"name": "letterbox_inverse", "pass": bool(ok)}


def test_boxes_in_bounds():
    """
    After clipping and filtering, all boxes should be valid and in image bounds.
    """
    w, h = 400, 300
    boxes = np.array([
        [-10, -5, 20, 25],     # partially out
        [380, 280, 450, 350],  # partially out
        [50, 50, 50, 80],      # zero width -> invalid
        [100, 100, 150, 160],  # valid
    ], dtype=np.float32)

    b = clip_boxes_xyxy(boxes, w, h)
    keep = valid_box_mask_xyxy(b)
    b = b[keep]

    cond = True
    if len(b):
        cond = (
            np.all(b[:, 0] >= 0)
            and np.all(b[:, 1] >= 0)
            and np.all(b[:, 2] <= w)
            and np.all(b[:, 3] <= h)
            and np.all((b[:, 2] - b[:, 0]) > 0)
            and np.all((b[:, 3] - b[:, 1]) > 0)
        )
    return {"name": "boxes_in_bounds", "pass": bool(cond)}


# ----------------------------
# Export reusable src/taco/data.py
# ----------------------------
def write_reusable_data_module(path: Path):
    """
    Writes a compact reusable module that Day 25 can import.
    (Mirrors the key dataset/geometry logic from this workflow script.)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    content = '''from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T


def coco_xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    out = boxes_xywh.copy().astype(np.float32)
    out[:, 2] = out[:, 0] + out[:, 2]
    out[:, 3] = out[:, 1] + out[:, 3]
    return out


def clip_boxes_xyxy(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    out = boxes.copy()
    out[:, 0] = np.clip(out[:, 0], 0, w)
    out[:, 1] = np.clip(out[:, 1], 0, h)
    out[:, 2] = np.clip(out[:, 2], 0, w)
    out[:, 3] = np.clip(out[:, 3], 0, h)
    return out


def valid_box_mask_xyxy(boxes: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    ww = boxes[:, 2] - boxes[:, 0]
    hh = boxes[:, 3] - boxes[:, 1]
    return (ww > eps) & (hh > eps)


@dataclass
class LetterboxMeta:
    scale: float
    pad_x: float
    pad_y: float
    src_w: int
    src_h: int
    dst_w: int
    dst_h: int


class DeterministicLetterbox:
    def __init__(self, out_w: int = 1024, out_h: int = 1024):
        self.out_w = int(out_w)
        self.out_h = int(out_h)

    def __call__(self, image: Image.Image, boxes_xyxy: np.ndarray):
        src_w, src_h = image.size
        scale = min(self.out_w / src_w, self.out_h / src_h)
        new_w = int(round(src_w * scale))
        new_h = int(round(src_h * scale))

        resized = image.resize((new_w, new_h), resample=Image.BILINEAR)

        pad_x = (self.out_w - new_w) / 2.0
        pad_y = (self.out_h - new_h) / 2.0

        left = int(np.floor(pad_x))
        top = int(np.floor(pad_y))
        right = self.out_w - new_w - left
        bottom = self.out_h - new_h - top

        padded = ImageOps.expand(resized, border=(left, top, right, bottom), fill=(114, 114, 114))

        boxes = boxes_xyxy.copy().astype(np.float32)
        if len(boxes):
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + left
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + top

        meta = LetterboxMeta(
            scale=float(scale),
            pad_x=float(left),
            pad_y=float(top),
            src_w=int(src_w),
            src_h=int(src_h),
            dst_w=self.out_w,
            dst_h=self.out_h,
        )
        return padded, boxes, meta


class TacoDetectionDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        annotations_json: Path,
        split_manifest_csv: Path,
        category_mapping_json: Path,
        split_name: str,
        out_size=(1024, 1024),
        train_mode=False,
        seed=42,
    ):
        self.data_dir = Path(data_dir)
        self.ann_file = Path(annotations_json)
        self.split_name = str(split_name)
        self.out_w, self.out_h = int(out_size[0]), int(out_size[1])
        self.train_mode = bool(train_mode)
        self.seed = int(seed)

        coco = json.loads(self.ann_file.read_text(encoding="utf-8"))
        self.images_df = pd.DataFrame(coco["images"])
        self.ann_df = pd.DataFrame(coco["annotations"])

        split_df = pd.read_csv(split_manifest_csv)
        split_df = split_df[split_df["split"] == self.split_name].copy()

        self.image_ids = split_df["image_id"].astype(int).tolist()
        self.images_df = self.images_df[self.images_df["id"].isin(self.image_ids)].copy()
        self.images_df = self.images_df.sort_values("id").reset_index(drop=True)

        mapping = json.loads(Path(category_mapping_json).read_text(encoding="utf-8"))
        self.orig_to_model = {
            int(k): int(v["model_category_id"])
            for k, v in mapping["original_to_model"].items()
        }

        ann_by_image = defaultdict(list)
        for rec in coco["annotations"]:
            ann_by_image[int(rec["image_id"])].append(rec)
        self.ann_by_image = ann_by_image

        self.letterbox = DeterministicLetterbox(out_w=self.out_w, out_h=self.out_h)
        self.rng = __import__("random").Random(self.seed)
        self.jitter = T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.02)

    def __len__(self):
        return len(self.images_df)

    def _apply_train_aug(self, image: Image.Image, boxes_xyxy: np.ndarray):
        if self.rng.random() < 0.5:
            w, _ = image.size
            image = ImageOps.mirror(image)
            if len(boxes_xyxy):
                x1 = boxes_xyxy[:, 0].copy()
                x2 = boxes_xyxy[:, 2].copy()
                boxes_xyxy[:, 0] = w - x2
                boxes_xyxy[:, 2] = w - x1

        image = self.jitter(image)
        return image, boxes_xyxy

    def __getitem__(self, idx):
        row = self.images_df.iloc[idx]
        image_id = int(row["id"])
        file_name = str(row["file_name"])
        img_path = (self.data_dir / file_name).resolve()

        try:
            with Image.open(img_path) as im:
                image = ImageOps.exif_transpose(im).convert("RGB")
        except (OSError, UnidentifiedImageError):
            image = Image.new("RGB", (int(row["width"]), int(row["height"])), (114, 114, 114))

        orig_w, orig_h = image.size
        anns = self.ann_by_image.get(image_id, [])

        boxes_xywh = []
        labels = []
        for a in anns:
            cat = int(a["category_id"])
            if cat not in self.orig_to_model:
                continue
            bbox = a.get("bbox", None)
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            x, y, w, h = [float(v) for v in bbox]
            if w <= 0 or h <= 0:
                continue
            boxes_xywh.append([x, y, w, h])
            labels.append(self.orig_to_model[cat])

        if len(boxes_xywh):
            boxes_xywh = np.array(boxes_xywh, dtype=np.float32)
            boxes_xyxy = coco_xywh_to_xyxy(boxes_xywh)
            boxes_xyxy = clip_boxes_xyxy(boxes_xyxy, orig_w, orig_h)
            keep = valid_box_mask_xyxy(boxes_xyxy)
            boxes_xyxy = boxes_xyxy[keep]
            labels = np.array(labels, dtype=np.int64)[keep].tolist()
        else:
            boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
            labels = []

        if self.train_mode:
            image, boxes_xyxy = self._apply_train_aug(image, boxes_xyxy)

        image_lb, boxes_lb, meta = self.letterbox(image, boxes_xyxy)
        boxes_lb = clip_boxes_xyxy(boxes_lb, self.out_w, self.out_h)
        keep2 = valid_box_mask_xyxy(boxes_lb)
        boxes_lb = boxes_lb[keep2]
        labels = np.array(labels, dtype=np.int64)[keep2].tolist()

        image_t = T.ToImage()(image_lb)
        image_t = T.ToDtype(torch.float32, scale=True)(image_t)

        target = {
            "boxes": torch.as_tensor(boxes_lb, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.as_tensor([image_id], dtype=torch.int64),
            "orig_size": torch.as_tensor([orig_h, orig_w], dtype=torch.int64),
            "size": torch.as_tensor([self.out_h, self.out_w], dtype=torch.int64),
            "transform_meta": {
                "scale": meta.scale, "pad_x": meta.pad_x, "pad_y": meta.pad_y,
                "src_w": meta.src_w, "src_h": meta.src_h,
                "dst_w": meta.dst_w, "dst_h": meta.dst_h,
            },
            "file_name": file_name,
        }
        return image_t, target
'''
    path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="TACO/data containing images + annotations.json")
    parser.add_argument(
        "--annotations-json",
        type=Path,
        default=None,
        help="Optional explicit path to annotations json. Default: <data-dir>/annotations.json",
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/day23"))
    parser.add_argument("--src-dir", type=Path, default=Path("src/taco"))
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--out-size", type=int, nargs=2, default=[1024, 1024], help="width height")
    parser.add_argument("--gallery-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = args.data_dir.expanduser().resolve()
    processed_dir = args.processed_dir.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    src_dir = args.src_dir.expanduser().resolve()
    config_dir = args.config_dir.expanduser().resolve()

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "geometry_gallery").mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    ann_path = args.annotations_json.expanduser().resolve() if args.annotations_json else (data_dir / "annotations.json")
    split_manifest = processed_dir / "split_manifest.csv"
    category_mapping = processed_dir / "category_mapping.json"

    if not ann_path.is_file():
        raise FileNotFoundError(f"Missing annotations json: {ann_path}")
    if not split_manifest.is_file():
        raise FileNotFoundError(f"Missing split manifest: {split_manifest}")
    if not category_mapping.is_file():
        raise FileNotFoundError(f"Missing category mapping: {category_mapping}")

    # Load mapping for label names
    mapping_obj = json.loads(category_mapping.read_text(encoding="utf-8"))
    model_name_map = {int(m["model_category_id"]): str(m["model_category_name"]) for m in mapping_obj["model_categories"]}

    out_w, out_h = int(args.out_size[0]), int(args.out_size[1])

    # Build datasets
    ds_train = TacoDetectionDataset(
        data_dir=data_dir,
        annotations_json=ann_path,
        split_manifest_csv=split_manifest,
        category_mapping_json=category_mapping,
        split_name="train",
        out_size=(out_w, out_h),
        train_mode=True,
        seed=args.seed,
    )
    ds_val = TacoDetectionDataset(
        data_dir=data_dir,
        annotations_json=ann_path,
        split_manifest_csv=split_manifest,
        category_mapping_json=category_mapping,
        split_name="val",
        out_size=(out_w, out_h),
        train_mode=False,
        seed=args.seed,
    )

    # ----------------------------
    # Run geometry tests
    # ----------------------------
    tests = [
        test_double_flip_identity(),
        test_letterbox_inverse(),
        test_boxes_in_bounds(),
    ]
    write_json(report_dir / "geometry_tests.json", {
        "all_pass": bool(all(t["pass"] for t in tests)),
        "tests": tests,
    })

    # ----------------------------
    # Collect quick dataset stats
    # ----------------------------
    def sample_stats(ds: TacoDetectionDataset, max_n: int = 500):
        n = min(len(ds), max_n)
        retained_boxes = 0
        imgs_with_zero_boxes = 0
        box_areas = []
        for i in range(n):
            _, tgt = ds[i]
            b = tgt["boxes"].numpy()
            retained_boxes += len(b)
            if len(b) == 0:
                imgs_with_zero_boxes += 1
            else:
                area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
                box_areas.extend(area.tolist())

        return {
            "sampled_images": int(n),
            "total_retained_boxes": int(retained_boxes),
            "images_with_zero_boxes": int(imgs_with_zero_boxes),
            "avg_boxes_per_image": float(retained_boxes / max(1, n)),
            "median_box_area_pixels": float(np.median(box_areas)) if box_areas else 0.0,
        }

    train_stats = sample_stats(ds_train, max_n=500)
    val_stats = sample_stats(ds_val, max_n=500)

    # ----------------------------
    # Save overlay gallery (val deterministic views)
    # ----------------------------
    rng = random.Random(args.seed)
    idx_pool = list(range(len(ds_val)))
    rng.shuffle(idx_pool)
    chosen = idx_pool[: min(args.gallery_size, len(idx_pool))]

    gallery_rows = []
    for i, idx in enumerate(chosen):
        # To get actual rendered image in transformed space, re-run internals on val item
        row = ds_val.images_df.iloc[idx]
        image_id = int(row["id"])
        file_name = str(row["file_name"])
        img_path = (data_dir / file_name).resolve()

        try:
            with Image.open(img_path) as im:
                image = ImageOps.exif_transpose(im).convert("RGB")
        except (OSError, UnidentifiedImageError):
            image = Image.new("RGB", (int(row["width"]), int(row["height"])), (114, 114, 114))

        anns = ds_val.ann_by_image.get(image_id, [])
        boxes_xywh = []
        labels = []
        for a in anns:
            cat = int(a["category_id"])
            if cat not in ds_val.orig_to_model:
                continue
            bb = a.get("bbox", None)
            if not isinstance(bb, (list, tuple)) or len(bb) != 4:
                continue
            x, y, w, h = [float(v) for v in bb]
            if w <= 0 or h <= 0:
                continue
            boxes_xywh.append([x, y, w, h])
            labels.append(ds_val.orig_to_model[cat])

        if len(boxes_xywh):
            b = coco_xywh_to_xyxy(np.array(boxes_xywh, dtype=np.float32))
            b = clip_boxes_xyxy(b, image.width, image.height)
            keep = valid_box_mask_xyxy(b)
            b = b[keep]
            labels = np.array(labels, dtype=np.int64)[keep].tolist()
        else:
            b = np.zeros((0, 4), dtype=np.float32)
            labels = []

        image_t, b_t, meta = ds_val.letterbox(image, b)
        b_t = clip_boxes_xyxy(b_t, out_w, out_h)
        keep2 = valid_box_mask_xyxy(b_t)
        b_t = b_t[keep2]
        labels = np.array(labels, dtype=np.int64)[keep2].tolist()

        over = draw_overlay(image_t, b_t, labels, model_name_map)
        out_fp = report_dir / "geometry_gallery" / f"val_{i:02d}_img{image_id}.jpg"
        over.save(out_fp, quality=90)

        gallery_rows.append({
            "gallery_index": i,
            "image_id": image_id,
            "file_name": file_name,
            "objects_drawn": int(len(labels)),
            "output_file": str(out_fp),
            "scale": meta.scale,
            "pad_x": meta.pad_x,
            "pad_y": meta.pad_y,
        })

    pd.DataFrame(gallery_rows).to_csv(report_dir / "geometry_gallery_manifest.csv", index=False)

    # ----------------------------
    # Write preprocessing contract
    # ----------------------------
    config = {
        "seed": int(args.seed),
        "task": "object_detection",
        "box_format_model": "xyxy_pixels",
        "source_box_format_coco": "xywh_pixels",
        "target_mapping_mode": mapping_obj["mode"],
        "out_size_wh": [out_w, out_h],
        "deterministic_resize": "letterbox_keep_aspect_ratio_with_padding",
        "train_augmentations": {
            "horizontal_flip_p": 0.5,
            "color_jitter": {
                "brightness": 0.15,
                "contrast": 0.15,
                "saturation": 0.05,
                "hue": 0.02,
            },
        },
        "val_augmentations": "none (deterministic preprocessing only)",
        "invalid_box_policy": "clip_to_bounds_then_drop_nonpositive_area",
        "empty_images_policy": "kept",
        "files": {
            "split_manifest": str(split_manifest),
            "category_mapping": str(category_mapping),
            "annotations_json": str(ann_path),
        },
    }
    write_json(config_dir / "preprocessing.json", config)

    # ----------------------------
    # Write src/taco/data.py
    # ----------------------------
    write_reusable_data_module(src_dir / "data.py")

    # ----------------------------
    # Write compact report
    # ----------------------------
    report = {
        "train_len": len(ds_train),
        "val_len": len(ds_val),
        "train_stats_sample": train_stats,
        "val_stats_sample": val_stats,
        "tests_all_pass": bool(all(t["pass"] for t in tests)),
        "tests": tests,
        "gallery_images_saved": len(gallery_rows),
        "artifacts": {
            "src_data_py": str(src_dir / "data.py"),
            "preprocessing_json": str(config_dir / "preprocessing.json"),
            "geometry_tests_json": str(report_dir / "geometry_tests.json"),
            "geometry_gallery_manifest": str(report_dir / "geometry_gallery_manifest.csv"),
        },
    }
    write_json(report_dir / "geometry_report.json", report)

    print("\nDay 23 completed.")
    print(f"- Wrote: {src_dir / 'data.py'}")
    print(f"- Wrote: {config_dir / 'preprocessing.json'}")
    print(f"- Wrote: {report_dir / 'geometry_tests.json'}")
    print(f"- Wrote: {report_dir / 'geometry_report.json'}")
    print(f"- Gallery: {report_dir / 'geometry_gallery'}")


if __name__ == "__main__":
    main()