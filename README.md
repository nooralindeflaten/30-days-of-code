# 30 Days of Machine Learning

A self-directed, 30-day Machine Learning challenge in the spirit of Advent of Code:
one focused, achievable-in-a-day challenge at a time, building up to six trained,
portfolio-ready models.

The point of this repo is to test **my** skills. Can I complete each challenge with
little to no AI help? The challenges are picked from a set of Claude-suggested topics,
but I don't get to see them in advance, and I don't get the solutions until I've had a
real attempt.

## Overview

### Challenges
Challenges are delivered in **batches** — typically one full model's worth (5 days)
at a time, like a sprint I can queue up and work through — rather than one-by-one.
Batches never come with the answers attached. The full roster of topics/models below
is public; the exact daily tasks aren't written down anywhere in this repo until I've
actually received and attempted them.

### Datasets
Each challenge either generates its own synthetic data (so nothing ever depends on a
flaky download) or points to a specific, listed public dataset/source.

### Code templates
Two different things share the word "template" in this repo, worth telling apart:

- **Starter files I get with a challenge** (rare) — occasional skeleton code needed
  to complete a task, included in `challenges/day-XX/` alongside `challenge.md`.
- **Beginner templates I create afterward** (see below) — a stripped-down version
  of *my own* finished solution, added back to `challenges/day-XX/` once I'm done,
  for anyone using `main` to attempt the challenge themselves from zero.

### Beginner-friendly templates
After I finish and commit my own solution for a day, I distill it into a starter
template: same file, function/class signatures and docstrings kept, actual logic
replaced with `TODO` comments. This gets added to `challenges/day-XX/` on `main`
(never before I've solved the day myself, so it can't act as a hint for me) —
it just means someone starting from a blank page later isn't staring at a truly
empty file.

```
challenges/day-03-preprocessing/
├── challenge.md
└── starter_template.py     ← added after I've solved it, for future beginners
```

### Reading material
`reading-material/` has a short, curated list of free resources for each of
the 6 models — for anyone who wants the background before
attempting a day cold. `00-start-here.md` is for complete beginners (math
prerequisites, Python/NumPy basics, what ML even is); `01` through `06` map
one-to-one to the 6 models. None of it contains challenge spoilers.

### File formats: `.py` vs `.ipynb`
Different days suit different formats — here's the rule of thumb I'm using:

| Kind of day | Format | Why |
|---|---|---|
| Implementing an algorithm from scratch (gradient descent, logistic regression, KNN, a decision tree, backprop, an RNN cell, etc.) | `.py` | You want clean, importable, testable code. Notebooks hide execution order and produce noisy, unreviewable git diffs for this kind of work. |
| Data exploration / preprocessing / feature engineering | `.ipynb` (or `.py`, either is fine) | Interleaving code, plots, and written observations as you go is exactly what notebooks are good at. |
| Model day (end of each 5-day block) | `.ipynb` recommended | You'll want plots and metrics inline for the model card and for screenshotting later — a notebook tells the story better than a script's stdout. |
| Deployment day (the Flask API) | `.py` | An API is an app, not a notebook. |

If in doubt, default to `.py` — it's easier for someone else (or future-me) to
run end-to-end and easier to diff in git. If you do use notebooks, clear cell
outputs before committing (`jupyter nbconvert --clear-output`) so diffs stay
readable — a notebook's saved outputs otherwise turn every commit into a wall
of noise.

### Structure: 6 models, 5-day cycles
Every 5 days ends with a trained model. The pattern repeats 6 times:

1. Four days of focused challenges (skills that feed into that cycle's model).
2. On day 5, I build, train, and evaluate a full model using what those four days
   taught me, and write it up like a real project (not just a script).
3. That model gets its own folder in `models/` with a short model card.
4. Repeat, next topic.

| Model | Days | Theme | Skills covered |
|---|---|---|---|
| 1 | 1–5 | Supervised Learning Baseline | gradient descent, classification/regression, metrics, cross-validation & tuning |
| 2 | 6–10 | Unsupervised Learning | K-Means, PCA, clustering evaluation, dimensionality reduction |
| 3 | 11–15 | Deep Learning / CNN | perceptron → MLP → backprop, intro to PyTorch, CNNs, image classification |
| 4 | 16–20 | NLP & Sequence Models | text preprocessing, embeddings, RNNs, attention basics |
| 5 | 21–25 | Computer Vision / Detection | augmentation, classical CV features, IoU/NMS, a from-scratch object detection pipeline |
| 6 | 26–30 | Reinforcement Learning + Deployment | multi-armed bandits, MDPs, Q-learning (Gymnasium), serving a model via an API |

## How to use this

1. Clone the repo (see **Repo structure & branches** below for which branch).
2. `pip install -r requirements.txt`
3. Ask for the next batch of challenges (typically a full model's 5 days at once).
4. Save each day's `challenge.md` into `challenges/day-XX/`.
5. Attempt each day myself, with as little AI assistance as possible, before
   looking anything up. Pick `.py` or `.ipynb` per the table above.
6. Write my solution into `my-solutions/day-XX/solution.py` (or `.ipynb`), plus
   a short `notes.md` in that folder (see "My solutions" below).
7. Once a day's solution is committed, distill a `starter_template.py` from it
   and add that to `challenges/day-XX/` on `main`.
8. On model days, save the trained model + writeup into `models/model-0N-<name>/`.
9. Post progress — a plot, a commit streak, a "here's what surprised me" note.

## Batch workflow (how challenges get revealed)
Because the whole point is not seeing challenges early, they aren't pre-written
and sitting in this repo waiting to be peeked at. Instead:
- I ask for "the next batch" (usually one model's worth of days).
- I get back that batch's `challenge.md` files — objective, tasks, hints — for
  every day in it. No solutions, no preview of the *next* batch.
- I don't get hints beyond what's in a challenge unless I ask for them after a
  real attempt at that specific day.
- I request a reference solution only after committing my own attempt for that
  day (or explicitly giving up on it).

## Repo structure & branches

This repo is split across two branches so someone can clone it and attempt
the challenges themselves without my solutions spoiling anything:

- **`main`** — the clean template: `README.md`, `requirements.txt`,
  `reading-material/`, and `challenges/` (challenge specs, added batch by
  batch, plus beginner `starter_template.py` files added in after I've solved
  each day — no solutions ever land here). This is the branch to clone if you
  want to run the challenge yourself.
- **`my-solutions`** — everything in `main`, plus `my-solutions/` (my actual
  attempts + honesty notes per day), `models/` (the 6 trained models + model
  cards), and a filled-in `PROGRESS.md`. This is my personal run of the
  challenge.

```bash
# Just want to try the challenges yourself?
git clone -b main <repo-url>

# Want to see how it went for me, solutions included?
git clone -b my-solutions <repo-url>
```

### How I keep the two in sync
A new batch of challenges gets added to `main` first, then merged into
`my-solutions` before I attempt any of it:

```bash
# on main: add the new batch of challenges
git checkout main
mkdir -p challenges/day-XX   # repeat for each day in the batch
# ...save each challenge.md (and any starter files) here...
git add challenges
git commit -m "Add Model N challenges (days XX-YY)"
git push origin main

# bring the batch into my working branch, then solve it there, one day at a time
git checkout my-solutions
git merge main
mkdir -p my-solutions/day-XX
# ...write solution.py/.ipynb + notes.md here, attempt the challenge...
git add my-solutions/day-XX
git commit -m "Day XX: my solution"

# then distill and add the beginner template back on main
git checkout main
# ...create challenges/day-XX/starter_template.py from the finished solution...
git add challenges/day-XX/starter_template.py
git commit -m "Day XX: add starter template"
git push origin main
git checkout my-solutions
git merge main
git push origin my-solutions
```

On model days, the trained model + model card also get committed to
`models/` on `my-solutions` only — `main` never carries any trained
artifacts, only challenge specs and starter templates.

## My solutions

My solutions are public in `my-solutions/`, and each one is tagged honestly:

- `notes.md` per day states: whether I used AI, and for what (debugging a
  specific error vs. writing the approach are very different); any code or
  concepts I pulled from elsewhere (Stack Overflow, docs, a blog post); how long
  it took; whether I hit the target unaided.
- Days marked **AI-assisted** in `notes.md` mean I got help beyond a quick syntax
  lookup — I'm not going to pretend those are unaided results.
- Model days may involve some AI-assisted review/debugging to make sure the final
  model actually works end-to-end — those are also tagged, and the model card
  says so.

The goal isn't a perfect streak. It's an honest one.

## Setup notes

- **Deep learning (Model 3 & 4):** PyTorch (CPU build is enough for everything here).
- **Object detection (Model 5):** kept conceptual/from-scratch — IoU, non-max
  suppression, and a simple detector on synthetic images/boxes. No large pretrained
  weights to download.
- **RL (Model 6):** [Gymnasium](https://gymnasium.farama.org/) for environments
  (FrozenLake/CartPole-scale, nothing exotic).
- Everything else runs on numpy/pandas/scikit-learn/matplotlib — no GPU required
  anywhere in this challenge.

## Progress

See [`PROGRESS.md`](PROGRESS.md) for a running log of days completed, time spent,
and whether each day was solved unaided.

## License

MIT — see `LICENSE`.