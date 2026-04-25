from project4.evaluation import *
from project4.parameters import *
from project4.maze import *
import matplotlib.pyplot as plt
from q_learn import *

# Some initialization #
eval_steps, eval_reward = [], []
learning = True

t = 0
pause_epochs = 50
total_epochs = NUM_STEPS
Q = np.zeros((NUM_STATES, NUM_ACTIONS))
env = Maze()
while t < total_epochs: # while learning
    # your Q-learning part goes here #
    current_Q_table = q_learning(env, Q=Q, numSteps=pause_epochs)

    avg_step, avg_reward = evaluation(Maze(), current_Q_table)
    eval_steps.append(avg_step)
    eval_reward.append(avg_reward)

    print(f"After {t} epochs: avg_step={avg_step}, avg_reward={avg_reward}")
    t += pause_epochs

# Plot example #
f1, [ax1, ax2] = plt.subplots(2, 1, figsize=(8,6))
# f1.suptitle("Performance Evaluation")

ax1.plot(np.arange(0,total_epochs,pause_epochs),eval_steps) # repeat for different algs.
ax1.set_ylabel("Average Number of Steps per Episode")
ax1.set_xlabel("Number of Epochs")

ax2.plot(np.arange(0,total_epochs,pause_epochs),eval_reward) # repeat for different algs.
ax2.set_ylabel("Average Cumulative Reward per Episode")
ax2.set_xlabel("Number of Epochs")

plt.tight_layout()
plt.savefig("02-q-learn/perf_eval_plot.png")
plt.show(block=True)