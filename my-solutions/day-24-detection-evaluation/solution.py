import numpy as np
import torch
from torchvision.ops import box_iou, batched_nms
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

# 1) IoU edge cases + comparison with torchvision
A = np.array([[0, 0, 10, 10]], float)
assert np.isclose(iou_matrix(A, A)[0, 0], 1.0)                                   # identical
assert iou_matrix(A, [[20, 20, 30, 30]])[0, 0] == 0                              # disjoint
assert np.isclose(iou_matrix(A, [[5, 5, 15, 15]])[0, 0], 25 / 175)               # partial
assert np.isclose(iou_matrix(A, [[2, 2, 8, 8]])[0, 0], 0.36)                     # containment
assert iou_matrix([[0, 0, 0, 10]], A)[0, 0] == 0                                 # zero area
B = np.random.default_rng(0).uniform(0, 100, (5, 2))
B = np.hstack([B, B + 20])
assert np.allclose(iou_matrix(B, B), box_iou(torch.tensor(B), torch.tensor(B)).numpy(), atol=1e-5)
print("IoU tests passed")

# 2) Matching demo: good hit, duplicate, wrong class, background false alarm
gt_b, gt_l = np.array([[10, 10, 60, 60], [100, 100, 160, 160]]), np.array([1, 2])
pr_b = np.array([[12, 12, 58, 58], [11, 11, 59, 59], [102, 102, 158, 158], [200, 200, 240, 240]])
pr_l, pr_s = np.array([1, 1, 1, 2]), np.array([0.95, 0.90, 0.80, 0.60])
order, status, fn = match(gt_b, gt_l, pr_b, pr_l, pr_s, iou_thr=0.5)
print("matching (score order):", status, "| missed GT:", fn.tolist())
# expected: ['tp', 'fp', 'fp', 'fp'] and GT #2 missed

# 3) Class-aware NMS demo (torchvision) - note NMS IoU is a separate setting from matching IoU
boxes = torch.tensor([[10, 10, 60, 60], [12, 12, 58, 58], [15, 15, 55, 55]], dtype=torch.float32)
scores, labels = torch.tensor([0.95, 0.90, 0.65]), torch.tensor([1, 1, 2])
print("NMS keeps indices:", batched_nms(boxes, scores, labels, 0.5).tolist())  # box 1 suppressed, box 2 survives (different class)

# 4) Frozen protocol (referenced by Days 25-26)
print("Protocol: xyxy pixels, class-aware matching at IoU 0.5, AP via pycocotools (Day 25).")