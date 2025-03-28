# Momentum Strategy
# Buy -> price increase over past k days
# Sell -> when price decreased over that interval

# Position : 0 & momentum_signal : 1 => Buy
# Position : 1 & momentum_signal : 0 => Sell

from .base_strat import BaseTradingEnv, random_policy
from ..main import off_policy_monte_carlo

class MomentumEnv(BaseTradingEnv):
    def __init__(self, episode_length=30, initial_price=100, k=5):
        self.k = k
        super().__init__(episode_length, initial_price)
    
    def get_state(self):
        if len(self.price_history) <= self.k:
            return (self.position, 0)
        momentum_signal = 1 if self.price_history[-1] > self.price_history[self.k - 1] else 0
        return (self.position, momentum_signal)
    
    def get_possible_states(self):
        return [(0,0), (0,1), (1,0), (1,1)] # Position, signal
    

def momentum_policy(state):
    position, momentum_signal = state
    if position == 0 and momentum_signal == 1:
        return {0: 0, 1: 1, 2: 0} # Buy
    elif position == 1 and momentum_signal == 0:
        return {0: 0, 1: 0, 2: 1} # Sell
    else:
        return {0: 1, 1: 0, 2: 0}
    
env = MomentumEnv(k=5)
pi = momentum_policy
b = random_policy

num_episodes = 10000
V = off_policy_monte_carlo(env, pi, b, num_episodes)
print("Use Case 2: Momentum Strategy")
for state in V:
    print(f"State {state}: V = {V[state]:.2f}")
