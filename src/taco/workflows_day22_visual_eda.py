"""
Day 22 — TACO visual EDA, target definition, and reproducible image-level split.

Lean scope:
- Data contract + core quality audit
- 4 purposeful plots
- Reproducible annotated gallery
- Target mapping (single-class litter default)
- Frozen split manifest for later days

Run:
  python -m workflows.day22_visual_eda --data-dir data/TACO
Optional:
  python -m src.taco.workflows_day22_visual_eda --data-dir ./data/TACO/ --output-dir data/TACO/reports/day22

Notes:
- Does NOT train any detector.
- Uses image-level split only (never annotation-level split).
- Raw data remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from sklearn.model_selection import train_test_split

SEED = 42


# ----------------------------
# Helpers
# ----------------------------
def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_plot(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def add_issue(issues: List[dict], issue: str, record_type: str, record_id, detail: str = ""):
    issues.append({
        "issue": str(issue),
        "record_type": str(record_type),
        "record_id": str(record_id),
        "detail": str(detail),
    })


def infer_split_stratify_label(
    img_ids: List[int],
    ann_by_image: Dict[int, List[dict]]
) -> Dict[int, str]:
    """
    Create a coarse image-level stratification label based on the dominant category per image.
    If an image has no annotations, label as '__empty__'.
    """
    labels = {}
    for iid in img_ids:
        anns = ann_by_image.get(iid, [])
        if not anns:
            labels[iid] = "__empty__"
            continue
        cats = [a["category_id"] for a in anns]
        dominant = Counter(cats).most_common(1)[0][0]
        labels[iid] = f"cat_{dominant}"
    return labels


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing TACO annotations.json and image files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/TACO/reports/day22"),
        help="Where to write Day 22 reports and plots",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Where to write split_manifest.csv and category_mapping.json",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation fraction for image-level split",
    )
    parser.add_argument(
        "--gallery-size",
        type=int,
        default=12,
        help="How many annotated gallery images to save",
    )
    parser.add_argument(
        "--missing-log",
        type=Path,
        default=None,
        help="Optional text file with missing-download image names from your downloader",
    )
    parser.add_argument(
        "--target-mode",
        choices=["single_class_litter", "multiclass_identity"],
        default="single_class_litter",
        help="Model category mapping style to save for later days",
    )
    args = parser.parse_args()

    if not (0 < args.val_fraction < 1):
        parser.error("--val-fraction must be between 0 and 1")

    data_dir = args.data_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    processed_dir = args.processed_dir.expanduser().resolve()

    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "gallery").mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    annotation_file = data_dir / "annotations.json"
    if not annotation_file.is_file():
        parser.error(f"Missing {annotation_file}")

    coco = json.loads(annotation_file.read_text(encoding="utf-8"))

    for k in ("images", "annotations", "categories"):
        if not isinstance(coco.get(k), list):
            raise ValueError(f"COCO json missing list '{k}'")

    images = pd.DataFrame(coco["images"])
    annotations = pd.DataFrame(coco["annotations"])
    categories = pd.DataFrame(coco["categories"])

    # Required columns
    for frame, req, name in (
        (images, ["id", "file_name", "width", "height"], "images"),
        (annotations, ["id", "image_id", "category_id", "bbox"], "annotations"),
        (categories, ["id", "name"], "categories"),
    ):
        missing = set(req) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")

    # ID uniqueness checks (report and continue where possible)
    issues = []
    if images["id"].duplicated().any():
        for rid in images.loc[images["id"].duplicated(keep=False), "id"].tolist():
            add_issue(issues, "duplicate_image_id", "image", rid)
    if categories["id"].duplicated().any():
        for rid in categories.loc[categories["id"].duplicated(keep=False), "id"].tolist():
            add_issue(issues, "duplicate_category_id", "category", rid)
    if annotations["id"].duplicated().any():
        for rid in annotations.loc[annotations["id"].duplicated(keep=False), "id"].tolist():
            add_issue(issues, "duplicate_annotation_id", "annotation", rid)

    # Category mapping (original)
    category_name_by_id = dict(zip(categories["id"], categories["name"]))
    category_id_set = set(categories["id"])

    # Optional missing log from downloader
    logged_missing = set()
    if args.missing_log is not None and args.missing_log.is_file():
        lines = [ln.strip() for ln in args.missing_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
        logged_missing = set(lines)

    # ----------------------------
    # Core image-level audit
    # ----------------------------
    image_meta = {int(r["id"]): r for _, r in images.iterrows()}
    image_ids = set(int(x) for x in images["id"].tolist())

    valid_image_ids = set()
    file_name_to_ids = defaultdict(list)

    for rec in coco["images"]:
        iid = int(rec["id"])
        file_name = str(rec["file_name"])
        file_name_to_ids[file_name].append(iid)

        img_path = (data_dir / file_name).resolve()
        # protect against path traversal in metadata
        try:
            if not img_path.is_relative_to(data_dir):
                add_issue(issues, "unsafe_image_path", "image", iid, file_name)
                continue
        except AttributeError:
            # Python < 3.9 fallback
            if str(data_dir) not in str(img_path):
                add_issue(issues, "unsafe_image_path", "image", iid, file_name)
                continue

        if not img_path.is_file():
            detail = "missing_on_disk"
            if file_name in logged_missing:
                detail = "missing_on_disk (also in downloader missing-log)"
            add_issue(issues, "missing_image", "image", iid, detail)
            continue

        try:
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im:
                w, h = im.size
            if int(rec["width"]) != int(w) or int(rec["height"]) != int(h):
                add_issue(
                    issues, "dimension_mismatch", "image", iid,
                    f"json={rec['width']}x{rec['height']}, file={w}x{h}"
                )
                continue
            if w <= 0 or h <= 0:
                add_issue(issues, "invalid_image_dimensions", "image", iid, f"{w}x{h}")
                continue

            valid_image_ids.add(iid)
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            add_issue(issues, "unreadable_image", "image", iid, repr(exc))

    for fname, ids in file_name_to_ids.items():
        if len(ids) > 1:
            for iid in ids:
                add_issue(issues, "duplicate_file_name", "image", iid, fname)

    # ----------------------------
    # Core annotation audit
    # ----------------------------
    valid_annotations = []
    for rec in coco["annotations"]:
        aid = int(rec["id"])
        iid = int(rec["image_id"])
        cid = int(rec["category_id"]) if str(rec["category_id"]).isdigit() else rec["category_id"]
        bbox = rec.get("bbox")

        errs = []

        if iid not in image_ids:
            errs.append("unknown_image_id")
        elif iid not in valid_image_ids:
            errs.append("unusable_image")

        if cid not in category_id_set:
            errs.append("unknown_category_id")

        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            errs.append("invalid_bbox_format")
        else:
            try:
                x, y, w, h = [float(v) for v in bbox]
                if not all(np.isfinite([x, y, w, h])):
                    errs.append("nonfinite_bbox")
                elif w <= 0 or h <= 0:
                    errs.append("nonpositive_bbox_size")
                elif iid in image_meta:
                    W = float(image_meta[iid]["width"])
                    H = float(image_meta[iid]["height"])
                    if x < 0 or y < 0 or x + w > W + 1e-6 or y + h > H + 1e-6:
                        errs.append("bbox_outside_image")
            except (ValueError, TypeError):
                errs.append("nonnumeric_bbox")

        if errs:
            for e in errs:
                add_issue(issues, e, "annotation", aid, f"image_id={iid}, category_id={cid}, bbox={bbox}")
        else:
            valid_annotations.append(rec)

    valid_ann_df = pd.DataFrame(valid_annotations)

    # Images with no annotations (flag only; do not force exclusion)
    ann_img_ids = set(int(x) for x in annotations["image_id"].tolist())
    for iid in sorted(image_ids - ann_img_ids):
        add_issue(issues, "image_without_annotations", "image", iid)

    issue_df = pd.DataFrame(issues, columns=["issue", "record_type", "record_id", "detail"])
    issue_df.to_csv(out / "data_quality_report.csv", index=False)

    # ----------------------------
    # Summary + mapping artifacts
    # ----------------------------
    category_counts = {}
    if not valid_ann_df.empty:
        tmp = valid_ann_df["category_id"].value_counts().to_dict()
        category_counts = {str(k): int(v) for k, v in tmp.items()}

    # Save mapping used by later days
    if args.target_mode == "single_class_litter":
        mapping = {
            "mode": "single_class_litter",
            "description": "All valid categories mapped to class_id=1 ('litter'). Background is implicit class_id=0.",
            "model_categories": [
                {"model_category_id": 1, "model_category_name": "litter"}
            ],
            "original_to_model": {
                str(cid): {"original_name": str(category_name_by_id[cid]), "model_category_id": 1}
                for cid in sorted(category_id_set)
            },
        }
    else:
        mapping = {
            "mode": "multiclass_identity",
            "description": "Original category IDs kept as model IDs.",
            "model_categories": [
                {"model_category_id": int(cid), "model_category_name": str(category_name_by_id[cid])}
                for cid in sorted(category_id_set)
            ],
            "original_to_model": {
                str(cid): {"original_name": str(category_name_by_id[cid]), "model_category_id": int(cid)}
                for cid in sorted(category_id_set)
            },
        }

    write_json(processed_dir / "category_mapping.json", mapping)

    summary = {
        "annotation_file": str(annotation_file),
        "seed": SEED,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "image_records": int(len(images)),
        "annotation_records": int(len(annotations)),
        "category_records": int(len(categories)),
        "usable_images": int(len(valid_image_ids)),
        "usable_annotations": int(len(valid_ann_df)),
        "issue_counts": issue_df["issue"].value_counts().to_dict(),
        "target_mode": args.target_mode,
        "category_counts_from_usable_annotations": category_counts,
    }
    write_json(out / "dataset_summary.json", summary)

    # ----------------------------
    # EDA plots (4)
    # ----------------------------
    # plot 1: category counts (usable anns)
    if not valid_ann_df.empty:
        cat_series = (
            valid_ann_df["category_id"].map(category_name_by_id)
            .value_counts()
            .sort_values()
        )
        plt.figure(figsize=(10, max(4, 0.25 * len(cat_series))))
        cat_series.plot.barh()
        plt.title("Category counts (usable annotations)")
        plt.xlabel("Count")
        save_plot(out / "plots" / "category_counts.png")

    # for image-level stats, only verified usable images
    usable_img = images[images["id"].isin(valid_image_ids)].copy()
    usable_img["aspect_ratio"] = usable_img["width"] / usable_img["height"]

    # object count per image
    img_obj_counts = Counter(valid_ann_df["image_id"]) if not valid_ann_df.empty else Counter()
    usable_img["object_count"] = usable_img["id"].map(lambda iid: img_obj_counts[int(iid)] if int(iid) in img_obj_counts else 0)

    # plot 2: objects per image
    plt.figure(figsize=(7, 4))
    bins = min(30, max(1, int(usable_img["object_count"].max()) + 1)) if len(usable_img) else 10
    plt.hist(usable_img["object_count"], bins=bins)
    plt.title("Objects per image (usable)")
    plt.xlabel("Annotation count")
    plt.ylabel("Images")
    save_plot(out / "plots" / "objects_per_image.png")

    # plot 3: image dimensions/aspect ratio
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(usable_img["width"], bins=30)
    axes[0].set_title("Image width distribution")
    axes[0].set_xlabel("Pixels")
    axes[0].set_ylabel("Images")

    axes[1].hist(usable_img["aspect_ratio"], bins=30)
    axes[1].set_title("Image aspect ratio distribution")
    axes[1].set_xlabel("width / height")
    axes[1].set_ylabel("Images")
    save_plot(out / "plots" / "image_geometry.png")

    # plot 4: relative box area
    if not valid_ann_df.empty:
        def rel_area(r):
            iid = int(r["image_id"])
            W = float(image_meta[iid]["width"])
            H = float(image_meta[iid]["height"])
            x, y, w, h = [float(v) for v in r["bbox"]]
            return (w * h) / (W * H)

        ra = valid_ann_df.apply(rel_area, axis=1).clip(lower=1e-12)
        plt.figure(figsize=(8, 4))
        plt.hist(np.log10(ra), bins=40)
        plt.title("Relative bounding-box area (log10)")
        plt.xlabel("log10(box_area / image_area)")
        plt.ylabel("Annotations")
        save_plot(out / "plots" / "relative_box_area_log10.png")

    # Save base tables
    usable_img[["id", "file_name", "width", "height", "aspect_ratio", "object_count"]].to_csv(
        out / "image_statistics.csv", index=False
    )
    if not valid_ann_df.empty:
        ann_stats = valid_ann_df.copy()
        ann_stats["category_name"] = ann_stats["category_id"].map(category_name_by_id)
        ann_stats.to_csv(out / "annotation_statistics.csv", index=False)

    # ----------------------------
    # Reproducible gallery (train split after split creation)
    # ----------------------------
    # Build ann lookup by image for overlays
    ann_by_image = defaultdict(list)
    for rec in valid_annotations:
        ann_by_image[int(rec["image_id"])].append(rec)

    # ----------------------------
    # Image-level split
    # ----------------------------
    split_candidates = sorted(valid_image_ids)
    if len(split_candidates) < 2:
        raise RuntimeError("Need at least 2 usable images for a split.")

    strat_labels_map = infer_split_stratify_label(split_candidates, ann_by_image)
    strat_labels = [strat_labels_map[iid] for iid in split_candidates]

    # If any class has <2 examples, sklearn stratify will fail. fallback to unstratified split.
    label_counts = Counter(strat_labels)
    can_stratify = min(label_counts.values()) >= 2 if label_counts else False

    if can_stratify:
        train_ids, val_ids = train_test_split(
            split_candidates,
            test_size=args.val_fraction,
            random_state=SEED,
            stratify=strat_labels,
        )
    else:
        train_ids, val_ids = train_test_split(
            split_candidates,
            test_size=args.val_fraction,
            random_state=SEED,
            stratify=None,
        )
        add_issue(
            issues, "split_not_stratified", "split", "global",
            "Some stratification labels had <2 samples; used unstratified split."
        )

    split_map = {iid: "train" for iid in train_ids}
    split_map.update({iid: "val" for iid in val_ids})

    manifest = images[["id", "file_name"]].copy().rename(columns={"id": "image_id"})
    manifest["split"] = manifest["image_id"].map(split_map).fillna("excluded_unusable")
    manifest.to_csv(processed_dir / "split_manifest.csv", index=False)

    split_summary = manifest["split"].value_counts().to_dict()
    write_json(out / "split_summary.json", split_summary)

    # Save category counts by split (usable annotations only)
    if not valid_ann_df.empty:
        split_ann = valid_ann_df[["image_id", "category_id"]].copy()
        split_ann["split"] = split_ann["image_id"].map(split_map)
        split_ann["category_name"] = split_ann["category_id"].map(category_name_by_id)
        split_cat = (
            split_ann.groupby(["split", "category_name"])
            .size()
            .unstack(fill_value=0)
        )
        split_cat.to_csv(out / "split_category_counts.csv")

    # ----------------------------
    # Gallery creation (train images only)
    # ----------------------------
    rng = random.Random(SEED)
    train_ids_sorted = sorted([iid for iid in train_ids if iid in valid_image_ids])

    # Half random + half crowded
    random_pick_n = min(args.gallery_size // 2, len(train_ids_sorted))
    selected = rng.sample(train_ids_sorted, random_pick_n) if random_pick_n > 0 else []

    crowded_sorted = sorted(train_ids_sorted, key=lambda iid: len(ann_by_image.get(iid, [])), reverse=True)
    for iid in crowded_sorted:
        if len(selected) >= args.gallery_size:
            break
        if iid not in selected:
            selected.append(iid)

    gallery_rows = []
    for iid in selected:
        rec = image_meta[iid]
        src = (data_dir / str(rec["file_name"])).resolve()

        try:
            with Image.open(src) as im:
                canvas = ImageOps.exif_transpose(im).convert("RGB")

            # If transpose changed dimensions vs annotation metadata, skip overlay and log
            if canvas.size != (int(rec["width"]), int(rec["height"])):
                add_issue(
                    issues, "exif_orientation_changes_dimensions", "image", iid,
                    f"file={rec['file_name']}, expected={rec['width']}x{rec['height']}, got={canvas.size}"
                )
                continue

            draw = ImageDraw.Draw(canvas)
            for a in ann_by_image.get(iid, []):
                x, y, w, h = map(float, a["bbox"])
                c_name = category_name_by_id.get(a["category_id"], str(a["category_id"]))
                draw.rectangle((x, y, x + w, y + h), outline="red", width=max(2, int(canvas.width / 500)))
                draw.text((x, max(0, y - 12)), c_name, fill="red")

            dst = out / "gallery" / f"image_{iid}.jpg"
            canvas.thumbnail((1200, 1200))
            canvas.save(dst, quality=88)

            gallery_rows.append({
                "image_id": iid,
                "source_file": rec["file_name"],
                "output_file": str(dst),
                "objects_drawn": len(ann_by_image.get(iid, [])),
            })
        except OSError as exc:
            add_issue(issues, "gallery_render_failure", "image", iid, repr(exc))

    pd.DataFrame(gallery_rows).to_csv(out / "gallery_manifest.csv", index=False)

    # rewrite issue report if new issues were added
    issue_df = pd.DataFrame(issues, columns=["issue", "record_type", "record_id", "detail"])
    issue_df.to_csv(out / "data_quality_report.csv", index=False)

    # ----------------------------
    # findings template (lean)
    # ----------------------------
    findings_template = (
        "# Day 22 findings\n\n"
        "## Three measured observations\n"
        "1. [Cite one CSV/plot value]\n"
        "2. [Cite one CSV/plot value]\n"
        "3. [Cite one CSV/plot value]\n\n"
        "## Three testable hypotheses for Day 25\n"
        "1. [Example: small objects dominate misses]\n"
        "2. [Example: dense scenes increase duplicate detections]\n"
        "3. [Example: rare categories are unstable under multiclass]\n\n"
        "## Decisions\n"
        f"- Annotation source: `{annotation_file}`\n"
        f"- Target mode: `{args.target_mode}`\n"
        "- Split: image-level, fixed seed, manifest saved in data/processed/split_manifest.csv\n"
    )
    (out / "findings.md").write_text(findings_template, encoding="utf-8")

    # final summary print
    print("\nSaved Day 22 outputs:")
    print(f"- quality report: {out / 'data_quality_report.csv'}")
    print(f"- plots: {out / 'plots'}")
    print(f"- gallery: {out / 'gallery'}")
    print(f"- split manifest: {processed_dir / 'split_manifest.csv'}")
    print(f"- category mapping: {processed_dir / 'category_mapping.json'}")
    print(f"- findings template: {out / 'findings.md'}")
    print("\nSplit summary:", split_summary)


if __name__ == "__main__":
    main()