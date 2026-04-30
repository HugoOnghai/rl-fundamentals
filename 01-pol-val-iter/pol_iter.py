from project4.maze import *
from project4.mdp import build_transitions
from project4.parameters import *
import numpy as np

# from lecture 18
# policy iteration (using iterative policy evaluation)
def policy_iteration(numStates: int, transitions):
    # initialization
    valueOfState = np.zeros(numStates) # V(s) in R for all s in S
    policyOfState = np.zeros(numStates) # pi(s) in A(s) for all s in S where A(s) denotes all possible actions from s

    policy_stable = False

    print("=== Beginning Policy Iteration ===")

    countIters = 0
    while(not policy_stable):
        countIters += 1

        # policy evaluation: valueOfState gets updated
        policy_evaluation(policyOfState, valueOfState, transitions)

        # policy improvement: policyOfState gets updated
        policy_stable, v_star, pi_star = policy_improvement(policyOfState, valueOfState, transitions)
        
    print(f"Policy Iteration stabilized after {countIters} iterations.")
    return v_star, pi_star # while loop breaks when policy is stable

def policy_evaluation(policy, valueOfState, transitions, gamma=GAMMA, theta=THETA):
    assert(len(policy) == len(valueOfState))
    numStates = len(policy)
    numActions = NUM_ACTIONS

    # pi'(s) = argmax_a sum_{s',r} p(s',r|s,a)[r + gamma v_pi(s')]
    delta = 1 # arbitrary value larger than theta
    while(delta > theta): # a small pos num determining the accuray of estim (from RL Book) break once delta < theta
        delta = 0

        for s in range(numStates):
            a = policy[s] # action determined optimal by policy
            v = valueOfState[s]

            # computes v_pi(s)
            valueOfState[s] = sum(
                joint(s_next,r,s,a,transitions) * (r+gamma*valueOfState[s_next]) 
                for s_next in range(numStates)
                for r in range(numActions)
            )
            delta = max(delta, np.abs(v - valueOfState[s]))


def policy_improvement(policy, valueOfState, transitions, gamma=GAMMA):
    assert(len(policy) == len(valueOfState))
    numStates = len(policy)
    numActions = NUM_ACTIONS

    # v_pi'(s) = max sum_{s',r} p(s',r|s,a)[r + gamma v_pi'(s')]
    policy_stable = True
    for s in range(numStates):
        old_action = policy[s]

        # q_pi(s,a), derived from/is Bellman's Equation
        def q_value(a): # only pass through a, s is defined already
            return sum(
                joint(s_next, r, s, a, transitions) * (r + gamma * valueOfState[s_next])
                for s_next in range(numStates)
                for r in range(numActions)
            )

        policy[s] = max(range(numActions), key=q_value)

        if old_action != policy[s]:
            policy_stable = False

    return policy_stable, valueOfState, policy # V approx v* and pi approx pi* if policy_stable

## helper functions
def joint(s_next, r, s, a, transitions):
    # p(s_next, r | s, a), where a is any action, not just pi(s)
    for triple in transitions[(s,a)]:
        if triple[1] == s_next and triple[2] == r:
            return triple[0] # probability of reaching s_next and observing reward r

    return 0 # if you cannot transition to s_next, prob = 0