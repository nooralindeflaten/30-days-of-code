import numpy as np


def choose_with_exploration_floor(greedy_action: int, n_arms: int, eps_floor: float, rng=None):
    """
    Epsilon-floor mixture:
      - with prob (1-eps_floor): choose greedy_action
      - with prob eps_floor: choose uniformly random arm

    Returns:
      (chosen_action, propensity_of_chosen_action)
    """
    rng = np.random.default_rng() if rng is None else rng
    greedy_action = int(greedy_action)
    n_arms = int(n_arms)
    eps_floor = float(eps_floor)

    if rng.random() < eps_floor:
        action = int(rng.integers(0, n_arms))
    else:
        action = greedy_action

    if action == greedy_action:
        propensity = (1.0 - eps_floor) + (eps_floor / n_arms)
    else:
        propensity = eps_floor / n_arms

    return action, float(propensity)


def fallback_action(default_action=0):
    return int(default_action), 1.0