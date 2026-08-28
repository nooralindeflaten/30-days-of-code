# Model 6 — Reinforcement Learning + Deployment: Reading Material

Covers: multi-armed bandits, MDPs, Q-learning, Gymnasium, serving a model via
an API.

## Reinforcement learning fundamentals
- **Spinning Up in Deep RL** (OpenAI): the standard free introduction to RL
  concepts and terminology — states, actions, policies, value functions —
  before touching any code. Still actively referenced despite being in
  "maintenance mode." https://spinningup.openai.com/
- **Sutton & Barto — Reinforcement Learning: An Introduction** (2nd edition,
  free official PDF from the authors): *the* RL textbook. Chapters 1–4 cover
  bandits and MDPs directly, which is exactly this model's first two days.
  http://incompleteideas.net/book/the-book-2nd.html

## Multi-armed bandits specifically
- **Lilian Weng — "The Multi-Armed Bandit Problem and Its Solutions"**: a
  widely-referenced, clear writeup of epsilon-greedy, UCB, and Thompson
  sampling. https://lilianweng.github.io/posts/2018-01-23-multi-armed-bandit/

## Gymnasium (the environments you'll actually train agents in)
- **Gymnasium official documentation**: https://gymnasium.farama.org/
- **Gymnasium — "Training an Agent" tutorial** (official): the standard
  Q-learning-on-FrozenLake walkthrough this model's challenges are inspired by.
  https://gymnasium.farama.org/introduction/train_agent/

## Deployment (shipping a model, not just training one)
- **Flask Quickstart** (official docs): everything you need for the `/predict`
  API endpoint. https://flask.palletsprojects.com/en/latest/quickstart/
- **Google — Machine Learning Crash Course: Production ML systems**: short,
  practical overview of what changes once a model has to serve real traffic.
  https://developers.google.com/machine-learning/crash-course/production-ml-systems
