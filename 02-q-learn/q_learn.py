import numpy as np
from project4.parameters import *
from project4.maze import *
import time

def q_learning(env, Q=None, numStates=NUM_STATES, numActions=NUM_ACTIONS, numSteps=NUM_STEPS, alpha=ALPHA, eps=EPSILON, gamma=GAMMA, Q_star=None, doPlot=True):
    if Q is None:
        # initialize Q-table and environment
        Q = np.zeros((numStates, numActions))
    # else, proceed with the Q provided

    s_t = env.reset()
    
    # Track RMSE values if Q_star is provided
    rmse_values = [] if Q_star is not None else None

    t = 0 # initialize the step counter

    while t < numSteps:
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
            print(f"On step {t} of {numSteps}: {(100*t/numSteps):.2f}%")
            time.sleep(SPP)

        # value iteration update
        Q[s_t, a_t] = Q[s_t, a_t] + alpha * (reward + gamma*max(Q[s_next]) - Q[s_t, a_t])
        
        # compute RMSE if Q_star is provided
        if Q_star is not None:
            rmse = np.sqrt(np.mean((Q - Q_star)**2))
            rmse_values.append(rmse)

        s_t = s_next

        if episodeDone:
            s_t = env.reset()

    # Return Q only, or (Q, rmse_values) based on whether Q_star was provided
    if Q_star is not None:
        return Q, rmse_values
    else:
        return Q

if __name__ == '__main__':
    env = Maze()
    
    Q = q_learning(env)
    print(Q)