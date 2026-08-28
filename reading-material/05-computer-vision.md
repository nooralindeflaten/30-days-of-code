# Model 5 — Computer Vision / Detection: Reading Material

Covers: image augmentation, classical CV features, IoU, non-max suppression,
object detection fundamentals.

## Classical computer vision (before it was all deep learning)
- **CS231n course notes**, the early lectures on image classification and
  linear classifiers give good grounding before jumping to detection:
  https://cs231n.github.io/
- **OpenCV-Python tutorials** (official docs) — edge detection, filtering,
  and other classical CV operations with runnable code:
  https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html

## Object detection concepts (IoU, NMS, anchor boxes)
- **Jonathan Hui — "mAP (mean Average Precision) for Object Detection"**
  and his companion posts on YOLO/R-CNN: widely referenced, clear
  explanations of IoU, NMS, and detection evaluation.
  https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
- **PyImageSearch — Intersection over Union (IoU) for object detection**:
  a very practical, code-first explanation of IoU specifically.
  https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

## Why data augmentation matters
- **PyTorch — Transforming and augmenting images** (official docs): the
  practical reference for how augmentation is actually implemented once
  you've built the from-scratch version yourself.
  https://docs.pytorch.org/vision/stable/transforms.html

## Going further (after this challenge)
- **Ultralytics YOLO docs**: once you understand IoU/NMS conceptually, this
  is the standard modern way to actually train/run object detectors.
  https://docs.ultralytics.com/
