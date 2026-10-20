# Day 29 — Sequential Decisions: MDPs and Q-Learning in Gymnasium

**Reinforcement Learning | Bellman updates, on-policy vs off-policy**

## Purpose
Learn the full RL setting, where actions change future states, and see why the routing problem is a bandit rather than a full RL problem.

## Business question
> When do decisions have lasting consequences, and what does it take to learn a policy that accounts for them?

## Inputs
- Gymnasium environments (`FrozenLake-v1` with `is_slippery=True` and `CliffWalking-v0`)

## Outputs (used by Day 31)
1. `src/rl/tabular.py` (Q-learning and SARSA, each about 25 lines)
2. `reports/day29/learning_curves.png` (return per episode, smoothed, multiple seeds)
3. `reports/day29/policy_grid.png` (greedy policy drawn on the CliffWalking grid)
4. `reports/day29/findings.md`

## Tasks
1. **Define the MDP** for each environment in a short table: states, actions, reward, discount, terminal conditions. Note which parts are stochastic.
2. **Implement tabular Q-learning** with epsilon decay. Train on FrozenLake over 5 seeds and plot the learning curve.
3. **Compare Q-learning and SARSA on CliffWalking.** Plot both learned paths and training returns.
4. **Explain the difference:** Q-learning learns the shorter path along the cliff edge, while SARSA learns the safer path, because the update targets differ.
5. **Gamma sensitivity:** run gamma = 0.5 and 0.99 on one environment.
6. **Write the bridge to Days 27–28:** a bandit is an MDP with one step, so state transitions don't matter. Say when routing would need full RL (for example when a user's history changes later requests).

## Deliverables
1. Both algorithms with learning curves over seeds
2. Policy visualization
3. MDP definition table
4. Short note linking bandits and RL

## Final questions
1. Why does Q-learning take the riskier path during training?
2. What do gamma and epsilon each control?
3. When is a bandit enough, and when do you need full RL?

## Time budget
**3–5 hours**

## Skip if short on time
The gamma sweep, and FrozenLake (keep CliffWalking, since it is the more insightful comparison).

## Constraint
Use Gymnasium's built-in environments. Don't build a custom one this week.