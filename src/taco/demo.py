import sys
import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor

path, thr = sys.argv[1], 0.5
model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)  # single-class: background + litter
model.load_state_dict(torch.load("data/TACO/models/detector.pt", map_location="cpu")["state_dict"]); model.eval()

try:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
except Exception as e:
    sys.exit(f"Could not read image: {e}")

with torch.no_grad():
    out = model([to_tensor(img)])[0]
d, n = ImageDraw.Draw(img), 0
for b, s in zip(out["boxes"], out["scores"]):
    if s >= thr:
        n += 1; d.rectangle(b.tolist(), outline="lime", width=3); d.text((b[0], max(0, b[1] - 12)), f"{s:.2f}", fill="lime")
img.save("demo_output.jpg")
print(f"{n} detections >= {thr}." + (" No detections does not mean the scene is litter-free." if n == 0 else ""))