# Model 3 — Deep Learning / CNN: Reading Material

Covers: perceptron, backpropagation, PyTorch basics, convolutional neural
networks, image classification.

## The single best resource for intuition
- **3Blue1Brown — Neural Networks** (YouTube playlist, 4 videos): builds up
  from "what is a neuron" through backpropagation with genuinely excellent
  visuals. Watch this before Day 1 of this model.
  https://www.youtube.com/playlist?list=PLZZWrBYkx7Otcjr3eCLZDCgfpqnxMY29s

## Backpropagation, explained properly
- Michael Nielsen's free online book, **Neural Networks and Deep Learning**,
  Ch. 2 (how backpropagation works, with the actual derivations):
  http://neuralnetworksanddeeplearning.com/chap2.html

## Learning PyTorch
- **PyTorch official "60 Minute Blitz" tutorial** — the standard starting
  point, covers tensors, autograd, and a full training loop:
  https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
- **PyTorch official tutorials index** (more depth once the blitz feels easy):
  https://docs.pytorch.org/tutorials/

## Convolutional Neural Networks
- **CS231n (Stanford) course notes** — the standard CV/CNN reference, used in
  countless university courses, completely free:
  https://cs231n.github.io/
- **DeepLearning.AI — Deep Learning Specialization**, Course 4
  (Convolutional Neural Networks), free to audit:
  https://www.coursera.org/specializations/deep-learning

## Why plain deep stacks of sigmoid/tanh don't train well
Relevant directly to the vanishing-gradient exercise in this model:
- CS231n notes on **Neural Networks Part 2: Setting up the data and the loss**
  (weight init, activation choice): https://cs231n.github.io/neural-networks-2/
