# Hugo Onghai
## Project 4, Intelligent Autonomous Systems

## 1. Policy Iteration

I chose to implement policy iteration. I instantiate `valueOfState` and `policyOfState`. `valueOfState` gets updated in `policy_evaluation(...)` and `policyOfState` gets updated in `policy_improvement(...)`. Once the policy is stable, `policyOfState` and `valueOfState` are returned as `pi_star` and `v_star`, respectively. Policy stability is determined state-by-state. If the updated policy prescribes a different action than the old policy for any state, the entire policy is considered not yet stable.

In `q_star.py` I build the optimal Q* table, based on `pi_star` and `v_star`. For any state and action pair `(s,a)`, `q_star` gives the value of that pair. So, for any state, the best action a* can be found as

`best_action = np.argmax(q_star[curr_state])`

We interpret Q* by plotting an episode which acts according to Q*. We print the final reward after the episode is done, i.e. when the goal state G is reached. To run it yourself, call this (assuming UV is installed):
```python
uv run 01-pol-val-iter/q_star.py
```

N.B., this will overwrite the current saved `01_q_star.npy`!

## 2. Q-learning

Here, I implemented Q-learning and experimented with different learning rates (`alpha`) and epsilons (for epsilon-greedy exploration vs. exploitation).

Q-learning runs through many episodes, exploring new and exploiting known reward pathways over time. The epsilon-greedy concept is applied to determine what probability of actions should be exploratory (random in nature) or exploitative (pick the best action at current state, as determined by Q). Over time, the reward after each step taken incrementally updates Q, which represents how reward propagates through the maze from the goal state G.

Parameters like learning rate and epsilon (and also the number of iterations) affect Q-learning performance. To track Q-learning performance, we compute the evolving RMSE of Q with respect to Q* obtained with policy iteration.

To run the parameter study:
```python
uv run 02-q-learn/parameter-study.py
```

To run the performance evaluations (evaluating the current policy after every 50 learning epochs):
```python
uv run 02-q-learn/perf_eval.py
```

N.B., these will overwrite the current `param_plot.png` and `perf_eval_plot.png`, respectively!

## 3. Continuous State Space Problems

To be implemented!