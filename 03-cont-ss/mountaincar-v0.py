# state vector is [position, velocity]
import gymnasium as gym
import matplotlib.pyplot as plt
from reinforce import *

n_episodes = NUM_EPISODES

# source: https://gymnasium.farama.org/environments/classic_control/mountain_car/
env = gym.make('MountainCar-v0', render_mode="rgb_array")

# 0 = accelerate left
# 1 = dont accelerate
# -1 = accelerate right
n_actions = 3 # discrete
state_dim = 2

theta, G_list = REINFORCE(env, n_actions, state_dim, render_env=False, n_episodes=n_episodes)

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
np.save("./03-cont-ss/03_theta_mountaincar", theta)