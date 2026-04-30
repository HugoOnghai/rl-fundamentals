from project4.parameters import *
import numpy as np
from gymnasium.wrappers import RecordVideo

# source for comments: my notes from "Reinforcement Learning: An Introduction" while I'm learning about the REINFORCE algo
# REINFORCE is a policy-gradient learning algo (pseudocode on page 328) that looks like stochastic gradient descent on policy

# input: a differentiable policy parameterization pi(a|s, theta)
# algorithm parameter, step size alpha > 0
def REINFORCE(env, n_actions, state_dim, render_env=True, max_steps=MAX_STEPS, n_episodes=NUM_EPISODES, alpha=ALPHA_REINFORCE, gamma=GAMMA_REINFORCE, record_every=RECORD_EVERY, cluster_every=5): # monte-carlo policy-gradient control (episodic) for pi_star
    # gymnasium has a record video feature, so i'll record every "record_every" episode
    env = RecordVideo(env, video_folder='03-cont-ss/mountaincar_videos', episode_trigger=lambda ep: ep % record_every == 0)
    
    print("=== Initializing REINFORCE ===")

    # initialize policy parameter theta in R^d' (e.g., to 0)
    theta = np.zeros((n_actions, state_dim))
    G_list = []

    # loop forever (for each episode):
    for ep in range(n_episodes):
        # Generate an episode S0,A0,R1,...,S_{T-1},A_{T-1},R_T, following pi(.|., theta)
        ep_state = []
        ep_action = []
        ep_reward = [None] # R_0 is DNE
        s_t, _ = env.reset()
        for t in range(max_steps):
            pi = compute_policy(s_t, theta)
            if t % cluster_every == 0: # clustering reduces frequency of changing action
                a_t = np.random.choice(n_actions, p=pi) # pick action from the probability distribution policy
            ep_state.append(s_t)
            ep_action.append(a_t)
            s_t, r_next, episodeDone, _, _ = env.step(a_t)
            ep_reward.append(r_next)

            if render_env:
                env.render()

            if episodeDone:
                print(f"Reached Goal State! Reward = {r_next}")
                break # don't take any more steps

        # Loop for each step of the episode t = 0,1,...,T-1;
        T = len(ep_reward) - 1
        G_t = np.zeros(T)
        G_t[T-1] = ep_reward[T]

        for t in range(T-2, -1, -1):
            G_t[t] = ep_reward[t+1] + gamma * G_t[t+1]

        G_list.append(G_t[0])
        # incorporate a baseline, just a simple moving average of total rewards
        baseline = np.mean(G_list)

        for t in range(T):
            delta = G_t[t] - baseline

            # theta_{t+1} = theta_{t} + alpha Gt ( grad(pi)/pi ) (this is the REINFORCE gradient update)
            # instead of grad(pi)/pi, i'll use log identity: grad w.r.t. theta of log(pi) = grad(pi)/pi, which is called the eligibility vector
            theta = theta + alpha * gamma**(t) * delta * compute_eligibility_vector(ep_state[t], ep_action[t], theta)

        print(f"Episode {ep}: T={T}, G0={G_t[0]:.2f}")

    return theta, G_list

def compute_policy(s, theta):
    h = theta.dot(s) # preferences are linear in features, where x is a feature vector describing the agent in the gymnasium environment (n_actions x state_dim)(state_dim,) = (n_actions,)
    h = h - np.max(h) # shifitng for numerical stability (numerator and denominator will cancel out)
    exp_h = np.exp(h)
    sum_exp_h = np.sum(exp_h) 
    return exp_h / sum_exp_h # exponential soft-max to get the policy's recommended action from highest preference from equation 13.2

def compute_eligibility_vector(s, a, theta): # obtain the eligibility vector from pi, "using the soft-max in action preference with linear action preferences"
    # grad(log(pi)) = x(s,a) - sum over b of (pi(b|s,theta)x(s,b)), where b is an arbitrary action from state s.
    # i'll just use my state vectors as my feature vectors, so x(s,a) = s for all a
    pi = compute_policy(s, theta)

    # pi (3,) s (6,). I had to format the equation 19 as an outer product to get it to work
    indicator = np.zeros(len(pi))
    indicator[a] = 1.0
    return np.outer(indicator - pi, s) # since sum over b of (pi*x) is the dot product (from equation 13.9)