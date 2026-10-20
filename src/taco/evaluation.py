import numpy as np


def iou_matrix(a, b):
    """Pairwise IoU between (N,4) and (M,4) boxes in xyxy format."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area = lambda x: np.clip(x[:, 2] - x[:, 0], 0, None) * np.clip(x[:, 3] - x[:, 1], 0, None)
    union = area(a)[:, None] + area(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def match(gt_boxes, gt_labels, boxes, labels, scores, iou_thr=0.5):
    """Greedy one-to-one, class-aware matching.
    Returns (order, status, fn_mask): status[k] is 'tp' or 'fp' for the k-th highest-scored
    prediction (order[k] is its original index); fn_mask marks unmatched GT boxes."""
    gt_boxes, boxes = np.asarray(gt_boxes).reshape(-1, 4), np.asarray(boxes).reshape(-1, 4)
    order = np.argsort(-np.asarray(scores))
    ious = iou_matrix(boxes[order], gt_boxes)
    taken = np.zeros(len(gt_boxes), bool)
    status = []
    for k, i in enumerate(order):
        cand = [j for j in range(len(gt_boxes))
                if not taken[j] and gt_labels[j] == labels[i] and ious[k, j] >= iou_thr]
        if cand:
            taken[max(cand, key=lambda j: ious[k, j])] = True
            status.append("tp")
        else:
            status.append("fp")
    return order, status, ~taken