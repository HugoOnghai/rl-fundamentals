from project4.parameters import *

def build_transitions(env):
    transitions = {} # 
    for s in range(env.snum):
        for a in range(env.anum):
            a_slip = (a+1)%4
            # compute the reward and next state after intended step
            r_nom, s_nom, _ = env.step(s,a)
            # compute the reward and next state after slip
            r_slip, s_slip, _ = env.step(s,a_slip)

            transitions[(s,a)] = [
                (P_NOM, s_nom, r_nom),
                (P_SLIP, s_slip, r_slip),
            ]

    return transitions