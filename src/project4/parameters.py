NUM_STATES = 112
NUM_ACTIONS = 4 # 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT

GAMMA = 0.9 # the discount factor
THETA = 1e-8 # small positive number determining the accuracy of estimation

P_NOM = 0.9 # prob of taking intended action
P_SLIP = 0.1 # prob of slipping to the next clockwise direction
assert(P_NOM + P_SLIP == 1)