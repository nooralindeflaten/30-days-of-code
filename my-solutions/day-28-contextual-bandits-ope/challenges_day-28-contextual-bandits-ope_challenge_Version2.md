# Day 28 — Context Matters: Contextual Bandits and Offline Evaluation

**Reinforcement Learning | Personalized decisions and counterfactual evaluation**

## Purpose
Make the decision depend on request features, then answer the question industry cares about most: **how do you evaluate a new policy using only logged data from an old one?**

## Business question
> Can a context-aware policy beat the best single option, and can we estimate that without deploying it?

## Inputs
- Day 27 environment, extended so each arm's payoff depends on a context vector (for example request difficulty and length)

## Outputs (used by Days 30 and 31)
1. `src/bandits/contextual.py` (LinUCB, or Thompson sampling with a logistic model)
2. `data/logged_interactions.csv` (context, action, **propensity**, reward, from a randomized logging policy)
3. `src/bandits/ope.py` (IPS and doubly robust estimators)
4. `reports/day28/ope_vs_truth.csv` (offline estimate vs the true value from the simulator)
5. `models/policy.joblib` (the chosen policy, frozen for Day 30)

## Tasks
1. **Context-dependent environment** where the best arm changes with context. Keep the true values available for checking.
2. **Logging policy:** collect data with a known randomized policy and save the action propensity in each row.
3. **Train LinUCB online** and compare it with the best fixed arm and with Day 27's non-contextual agent.
4. **Offline evaluation:** estimate the new policy's value from the log using IPS and doubly robust, and compare against the true simulator value.
5. **Note the failure mode:** try a logging policy with very low propensities for some actions, then report the estimator variance.

## Deliverables
1. Contextual agent and learning curve
2. Logged dataset with propensities
3. Offline estimates compared against ground truth
4. Short note on when offline evaluation can't be trusted

## Final questions
1. How much does context add over the best fixed arm?
2. How close did IPS and DR come to the true value, and why do they differ in variance?
3. Why must the log store propensities?

## Time budget
**4–5 hours**

## Skip if short on time
A second contextual algorithm, and the low-propensity experiment.

## Constraint
Offline estimates are compared against simulator truth only. Don't call them proof of real-world performance.