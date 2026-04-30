NUM_STATES = 112
NUM_ACTIONS = 4 # 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT

GAMMA = 0.9 # the discount factor
THETA = 1e-4 # small positive number determining the accuracy of estimation

P_NOM = 0.9 # prob of taking intended action
P_SLIP = 0.1 # prob of slipping to the next clockwise direction
assert(P_NOM + P_SLIP == 1)

## Q-Learning
ALPHA = 0.5 # step size
EPSILON = 0.5 # epsilon for exploitation vs. exploration
Q_NUM_EPISODES = 1000
MAX_STEPS = 50

## MAZE
SPP = 0.5 # seconds per plot

## REINFORCE
GAMMA_REINFORCE = 1 - 10**-4
ALPHA_REINFORCE = 0.001
NUM_EPISODES = 5000
MAX_STEPS = 1000
RECORD_EVERY = 1000