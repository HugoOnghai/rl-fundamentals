import numpy as np
from project4.parameters import *
from project4.maze import *
import time

def q_learning(env, Q=None, num_episodes=Q_NUM_EPISODES, max_steps=MAX_STEPS, numActions=NUM_ACTIONS, numStates=NUM_STATES, alpha=ALPHA, eps=EPSILON, gamma=GAMMA, Q_star=None, doPlot=True):
    if Q is None:
        # initialize Q-table and environment
        Q = np.zeros((numStates, numActions))
    # else, proceed with the Q provided
    
    # Track RMSE values if Q_star is provided
    rmse_values = [] if Q_star is not None else None

    for ep in range(num_episodes):
        s_t = env.reset()
        t = 0 # initialize the step counter

        while t < max_steps:
            # choose to explore or exploit based on epsilon-defined probability
            doExploit = np.random.choice([True, False], p=[1-eps, eps])

            # epsilon-greedy exploration/exploitation
            if doExploit:
                a_t = int(np.argmax(Q[s_t])) # greedy action
                reward, s_next, episodeDone = env.step(s_t, a_t)
            else:
                a_t = int(np.random.choice(4)) # explore randomly action
                reward, s_next, episodeDone = env.step(s_t, a_t)

            t += 1
            if not doPlot:
                env.plot(s_next, a_t, extra_lines=1)
                print(f"On step {t} of {max_steps}: {(100*t/max_steps):.2f}%")
                time.sleep(SPP)

            # value iteration update
            Q[s_t, a_t] = Q[s_t, a_t] + alpha * (reward + gamma*max(Q[s_next]) - Q[s_t, a_t])

            s_t = s_next

            if episodeDone:
                # compute RMSE if Q_star is provided
                if Q_star is not None:
                    rmse = np.sqrt(np.mean((Q - Q_star)**2))
                    rmse_values.append(rmse) # only append last value of rmse

                break

    # Return Q only, or (Q, rmse_values) based on whether Q_star was provided
    if Q_star is not None:
        return Q, rmse_values
    else:
        return Q

if __name__ == '__main__':
    env = Maze()
    
    Q = q_learning(env)
    print(Q)