import numpy as np
from project4.parameters import *
from pol_iter import *
import time

def construct_q_star(v_star, pi_star, transitions, gamma=GAMMA):
    assert(len(pi_star) == len(v_star))
    numStates = len(pi_star)
    numActions = NUM_ACTIONS

    q_star = np.zeros((numStates, numActions))
    
    for s in range(numStates):
        for a in range(numActions):
            q_star[s][a] = sum(
                joint(s_next, r, s, a, transitions) * (r + gamma * v_star[s_next])
                for s_next in range(numStates)
                for r in range(numActions)
            ) 
            # max of q(s_next, any a) = v_star(s_next)
            # that is, the value of a state is its pairing with the best action 

    return q_star

def save_q_star(q_star):
    np.savez("./01-pol-val-iter/01_q_star", q_star)

def interpret_q_star(env, q_star):
    curr_state = env.reset() # start in the top left with no flags collected
    print("\n" * (env.dim[0]+1), end="")
    env.plot(curr_state, None, extra_lines=1)

    done = False
    while(not done):
        best_action = np.argmax(q_star[curr_state])
        reward, curr_state, done = env.step(curr_state, best_action) # update curr_state after taking best action
        print("Current reward: %s" % reward)
        time.sleep(0.75)
        env.plot(curr_state, best_action, extra_lines=2)

    print("Current reward: %s" % reward)

if __name__ == '__main__':
    env = Maze()
    P = build_transitions(env)

    v_star, pi_star = policy_iteration(NUM_STATES, P)

    print("=== Constructing & Saving Q*(s,a) ===")
    q_star = construct_q_star(v_star, pi_star, P)
    save_q_star(q_star)

    print("=== Optimal Path from Q*(s,a) ===")
    interpret_q_star(env, q_star)