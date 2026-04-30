import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from project4.maze import *
from q_learn import *

plt.rcParams['text.usetex'] = True

# load Q* from policy iteration 01-pol-val-iter/q_star.py
Q_star = np.load('01-pol-val-iter/01_q_star.npy')

# initialize array of epsilons...
epsilons = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# run Q-Learning with each epsilon
fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(8,6))
env = Maze()

# create colormap and normalize epsilon values to [0, 1]
cmap = plt.get_cmap('viridis')
norm = mcolors.Normalize(vmin=min(epsilons), vmax=max(epsilons))
sm = cm.ScalarMappable(cmap=cmap, norm=norm)

for eps in epsilons:
    Q, rmse_values = q_learning(env, eps=eps, Q_star=Q_star)
    color = sm.to_rgba(eps)
    ax1.plot(rmse_values, color=color, label=f'{eps}')

# create colormap and normalize epsilon values to [0, 1]
cmap = plt.get_cmap('inferno')
norm = mcolors.Normalize(vmin=min(alphas), vmax=max(alphas))
sm = cm.ScalarMappable(cmap=cmap, norm=norm)

for a in alphas:
    Q, rmse_values = q_learning(env, alpha=a, Q_star=Q_star)
    color = sm.to_rgba(a)
    ax2.plot(rmse_values, color=color, label=f'{a}')

fig.suptitle("Optimal Q Values with Hyperparameter Tuning")
ax1.set_ylabel(r"RMSE (w.r.t. $Q^\star$)")
ax1.set_xlabel("Episode")
ax1.set_title(rf"Tuning Epsilon (fixed $\alpha=${ALPHA})")
ax1.legend(title=r"$\epsilon$")

ax2.set_ylabel(r"RMSE (w.r.t. $Q^\star$)")
ax2.set_xlabel("Episode")
ax2.set_title(rf"Tuning Alpha (fixed $\epsilon=${EPSILON})")
ax2.legend(title=r"$\alpha$")

plt.tight_layout()
plt.savefig("02-q-learn/param_plot.png")
plt.show(block=True)