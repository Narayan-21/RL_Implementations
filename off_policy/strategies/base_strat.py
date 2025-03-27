import numpy as np


class BaseTradingEnv:
    def __init__(self, episode_length=30, initial_price = 100):
        self.episode_length = episode_length
        self.initial_price = initial_price
        self.actions = {0: "hold", 1: "buy", 2: "sell"}
        self.reset()
        
    def reset(self):
        self.t = 0
        self.position = 0 # 0: No stock, 1: Stock holder
        self.purchase_price = None
        self.price_history = [self.initial_price]
        # Generate Initial Prices
        for _ in range(20):
            self.price_history.append(self.price_history[-1] + np.random.normal(0,1))
        self.state = self.get_state()
        return self.state
    
    def step(self, action):
        reward = 0
        current_price = self.price_history[-1]
        if action == 1 and self.position == 0:
            self.position = 1
            self.purchase_price = current_price
        elif action == 2 and self.position == 1:
            self.position = 0
            reward = current_price - self.purchase_price
        
        # Generate Next price
        next_price = current_price + np.random.normal(0, 1)
        self.price_history.append(next_price)
        self.t += 1
        done = (self.t >= self.episode_length)
        next_state = self.get_state()
        return next_state, reward, done, {}
    
    def get_state(self):
        raise NotImplementedError("SubClass must implement get_state()")
    
    def get_possible_states(self):
        raise NotImplementedError("SubClass must implement get_possible_states()")