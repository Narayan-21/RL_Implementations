# Multiple Indicator Strategy
# MA Crossover + momentum signal
# position = 0 and both ma_signal and momentum_signal= 1 => Buy
# position = 1 and both ma_signal and momentum_signal= 0 => Sell
import numpy as np
from .base_strat import BaseTradingEnv, random_policy
from ..main import off_policy_monte_carlo

class MultiIndicator(BaseTradingEnv):
    def __init__(self, episode_length=30, initial_price=100, momentum_k=5, short_window=5, long_window = 10):
        self.momentum_k = momentum_k
        self.short_window = short_window
        self.long_window = long_window
        super().__init__(episode_length, initial_price)
        
    def get_state(self):
        if len(self.price_history) < max(self.long_window, self.momentum_k):
            return (self.position, 0, 0)
        short_MA = np.mean(self.price_history[-self.short_window:])
        long_MA = np.mean(self.price_history[-self.long_window:])
        MA_signal = 1 if short_MA > long_MA else 0
        momentum_signal = 1 if self.price_history[-1] > self.price_history[-self.momentum_k - 1] else 0
        return (self.position, MA_signal, momentum_signal)
    
    def get_possible_states(self):
        return [(p, ma, mom) for p in (0, 1) for ma in (0, 1) for mom in (0, 1)]
    
def multi_indicator_policy(state):
    position, MA_signal, momentum_signal = state
    if position == 0 and MA_signal == 1 and momentum_signal == 1:
        return {0: 0, 1: 1, 2: 0} # Buy
    elif position == 1 and MA_signal == 0 and momentum_signal == 0:
        return {0: 0, 1: 0, 2: 1} # Sell
    else:
        return {0: 1, 1: 0, 2: 0}  # Hold\\

env = MultiIndicator()
pi = multi_indicator_policy
b = random_policy
num_episodes = 100000
V = off_policy_monte_carlo(env, pi, b, num_episodes)
print("Use Case 4: Multiple Indicators Strategy")
for state in V:
    print(f"State {state}: V = {V[state]:.2f}")