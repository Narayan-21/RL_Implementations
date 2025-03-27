# Moving Average Trading Strategy -
# If short term MA > Long term MA => Buy
# elif short term MA < long term MA => Sell

import numpy as np
from .base_strat import BaseTradingEnv
from ..main import off_policy_monte_carlo

class MAEnv(BaseTradingEnv):
    def __init__(self, episode_length=30, initial_price=100, short_window=5, long_window=20):
        self.short_window = short_window
        self.long_window = long_window
        super().__init__(episode_length, initial_price)
    
    def get_state(self):
        if len(self.price_history) < self.long_window:
            return (self.position, 0) # Default until enough data
        short_MA = np.mean(self.price_history[-self.short_window:])
        long_MA = np.mean(self.price_history[-self.long_window:])
        MA_signal = 1 if short_MA > long_MA else 0
        return (self.position, MA_signal)
    
    def get_possible_states(self):
        return [(0,0), (0,1), (1,0), (1,1)] # (position => position exists: 1 / or not: 0, MA_signal => short_ma > long_ma : 1 else 0)
    
    
def ma_policy(state):
    position, MA_signal = state
    if position == 0 and MA_signal == 1:
        return {0: 0.0, 1: 1.0, 2: 0.0} # Buy 
    elif position == 1 and MA_signal == 0:
        return {0: 0.0, 1: 0.0, 2: 1.0} # Sell 
    else:
        return {0: 1.0, 1: 0.0, 2: 0.0} # Hold
    
def random_policy(state):
    return {0: 1/3, 1: 1/3, 2: 1/3}

if __name__ == "__main__":
    env = MAEnv()
    pi = ma_policy
    b = random_policy
    num_episodes = 10000
    V = off_policy_monte_carlo(env, pi, b, num_episodes)

    print("Use Case 1: Moving Average Crossover Strategy")
    for state in V:
        print(f"State {state}: V = {V[state]:.2f}")