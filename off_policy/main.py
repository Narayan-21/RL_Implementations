# Off policy learning
import numpy as np

def generate_episode(env, policy):
    states, actions, rewards = [], [], []
    state = env.reset()
    DONE = False
    while not DONE:
        action_prob = policy(state)
        action = np.random.choice(list(action_prob.keys()), p=list(action_prob.values()))
        next_state, reward, DONE, _ = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
    return states, actions, rewards

def off_policy_monte_carlo(env, pi, b, num_episodes, gamma=1.0):
    """
    Off-policy Monte Carlo policy evaluation.
    
    Args:
        env: Trading environment with reset() and step() methods.
        pi: Target policy (dict mapping state to action probabilities).
        b: Behavior policy (dict mapping state to action probabilities).
        num_episodes: Number of episodes to simulate.
        gamma: Discount factor (default 1.0 for undiscounted returns).
    
    Returns:
        V: Estimated value function (dict mapping states to values).
    """
    possible_states = env.get_possible_states()
    V = {state: 0.0 for state in possible_states}
    W = {state: 0.0 for state in possible_states}
    C = {state: 0.0 for state in possible_states}
    
    for _ in range(num_episodes):
        states, actions, rewards = generate_episode(env, b)
        T = len(states)
        G = 0.0
        rho_cum = 1.0 # Cumulative importance sampling ratio
        
        for t in range(T-1, -1, -1):
            state = states[t]
            action = actions[t]
            G = rewards[t] + gamma * G
            pi_prob = pi(state).get(action, 0.0)
            b_prob = b(state).get(action, 0.0)
            rho_t = pi_prob / b_prob if b_prob > 0 else 0
            rho_cum *= rho_t
            
            W[state] += rho_cum
            C[state] += rho_cum * G
            if W[state] > 0:
                V[state] = C[state] / W[state]
    
    return V


        
        