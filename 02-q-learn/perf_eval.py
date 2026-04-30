from project4.evaluation import *
from project4.parameters import *
from project4.maze import *
import matplotlib.pyplot as plt
from q_learn import *

plt.rcParams['text.usetex'] = True

pause_episodes = 10
total_episodes = Q_NUM_EPISODES
epochs = np.arange(0, total_episodes, pause_episodes)

env = Maze()

# (alpha, epsilon) combinations
param_combos = [
    (0.1, 0.1),
    (0.1, 0.2),
    (0.05, 0.1),
    (0.2, 0.3),
    (0.3, 0.4),
    (0.3, 1.0)
]

fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(8,6))

for alpha, eps in param_combos:
    print(f"Now running Q-learning for {(alpha, eps)}")
    eval_steps, eval_reward = [], []
    Q = np.zeros((NUM_STATES, NUM_ACTIONS))
    t = 0
    while t < total_episodes:
        q_learning(env, Q=Q, num_episodes=pause_episodes, max_steps=MAX_STEPS, alpha=alpha, eps=eps)
        avg_step, avg_reward = evaluation(Maze(), Q)
        eval_steps.append(avg_step)
        eval_reward.append(avg_reward)
        t += pause_episodes
    ax1.plot(epochs, eval_steps, label=rf"$\alpha$={alpha}, $\epsilon$={eps}")
    ax2.plot(epochs, eval_reward, label=rf"$\alpha$={alpha}, $\epsilon$={eps}")

ax1.set_ylabel("Average Number of Steps per Episode")
ax1.set_xlabel("Number of Episodes")
ax1.legend()

ax2.set_ylabel("Average Cumulative Reward per Episode")
ax2.set_xlabel("Number of Episodes")
ax2.legend()

fig.suptitle("Performance Evaluation")

plt.tight_layout()
plt.savefig(f"02-q-learn/perf_eval_plot.png")
plt.show(block=True)
