# state vector is [cos(theta1), sin(theta1), cos(theta2), sin(theta2), theta_dot1, theta_dot2]
import gymnasium as gym
import matplotlib.pyplot as plt
from reinforce import *

n_episodes = NUM_EPISODES

# source: https://gymnasium.farama.org/environments/classic_control/acrobot/
env = gym.make('Acrobot-v1', render_mode="rgb_array")

# 0 = apply -1 torque
# 1 = apply 0 torque
# -1 = apply 1 torque
n_actions = 3 # discrete
state_dim = 6

theta, G_list = REINFORCE(env, n_actions, state_dim, render_env=False, n_episodes=n_episodes, cluster_every=1)

window = 50
smoothed = np.convolve(G_list, np.ones(window)/window, mode='valid')

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(range(len(smoothed)), smoothed, label=f'{window}-episode moving average')
ax.plot(range(len(G_list)), G_list, alpha=0.3, label='raw G0')
ax.set_xlabel('Episode')
ax.set_ylabel('G0')
ax.legend()

plt.tight_layout()
plt.savefig("03-cont-ss/acrobot-totalreward.png")
plt.show(block=True)

# save theta
np.save("./03-cont-ss/03_theta", theta)