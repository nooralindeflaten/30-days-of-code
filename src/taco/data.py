from __future__ import annotations

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
