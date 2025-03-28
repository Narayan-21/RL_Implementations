# Mean reversal - buy when price below the historical mean else sell
import numpy as np
from .base_strat import BaseTradingEnv, random_policy
from ..main import off_policy_monte_carlo

class MeanReversal(BaseTradingEnv):
    def __init__(self, episode_length=30, initial_price=100, window = 10):
        self.window = window
        super().__init__(episode_length, initial_price)
        
    def get_state(self):
        if len(self.price_history) <= self.window:
            return (self.position, 0)
        mean_price = np.mean(self.price_history[-self.window - 1 :-1])
        price_level = 0 if self.price_history[-1] < mean_price else 1
        return (self.position, price_level)
    
    def get_possible_states(self):
        return [(0,0), (0,1), (1,0), (1,1)]


def mean_reversal_policy(state):
    position, price_level = state
    if position == 0 and price_level == 0:
        return {0: 0, 1: 1, 2: 0} # Buy
    elif position == 1 and price_level == 1:
        return {0: 0, 1: 0, 2: 1} # Sell
    else:
        return {0: 1, 1: 0, 2: 0} # Hold
    
env = MeanReversal(window=10)
pi = mean_reversal_policy
b = random_policy
num_episodes = 10000
V = off_policy_monte_carlo(env, pi, b, num_episodes)
print("Use Case 3: Mean-Reversion Strategy")
for state in V:
    print(f"State {state}: V = {V[state]:.2f}")
