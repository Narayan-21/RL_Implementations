import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Tuple
import gymnasium as gym

class MonteCarlo:
    def __init__(self, action_space: int, gamma: float=0.9, epsilon: float=0.1):
        self.action_space: int = action_space
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.Q: Dict[Any, np.ndarray] = defaultdict(lambda: np.zeros(action_space))  # Action-value function
        self.N: Dict[Any, np.ndarray] = defaultdict(lambda: np.zeros(action_space))  # Count visits N(s,a)

    def generate_episode(self, env: gym.Env) -> List[Tuple[Any, int, float]]:
        episode = []
        state, _ = env.reset()
        terminated = truncated = False

        while not (terminated or truncated):
            action: int = self.epsilon_greedy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
        return episode

    def epsilon_greedy(self, state: Any) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space)
        return int(np.argmax(self.Q[state]))

    def update_q_values(self, episode: List[Tuple[Any, int, float]]) -> None:
        G: float = 0
        visited: set = set()
        for i in reversed(range(len(episode))):
            state, action, reward = episode[i]
            G = reward + self.gamma * G
            if (state, action) not in visited:
                visited.add((state, action))
                self.N[state][action] += 1
                alpha: float = 1 / self.N[state][action]
                self.Q[state][action] += alpha * (G - self.Q[state][action])

    def train(self, env: gym.Env, episodes: int = 10000) -> None:
        for episode_num in range(episodes):
            episode = self.generate_episode(env)
            self.update_q_values(episode)
            if (episode_num + 1) % 1000 == 0:
                print(f"Completed {episode_num + 1} episodes")

    def get_policy(self):
        policy: Dict[Any, int] = {state: int(np.argmax(actions)) for state, actions in self.Q.items()}
        print("policy ->", policy)
        return policy
    
    def print_policy_grid(self, shape=(4, 4)):
        policy = self.get_policy()
        
        action_symbols = {
            0: "←",  # LEFT
            1: "↓",  # DOWN
            2: "→",  # RIGHT
            3: "↑"   # UP
        }
        
        print("Policy Grid:")
        for i in range(shape[0]):
            row = ""
            for j in range(shape[1]):
                state_idx = i * shape[1] + j
                if state_idx in policy:
                    row += action_symbols[policy[state_idx]] + " "
                else:
                    row += "? "
            print(row)


if __name__ == "__main__":
    
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode = "rgb_array")
    
    agent = MonteCarlo(action_space=env.action_space.n)
    
    agent.train(env=env, episodes=5000)
    
    policy = agent.get_policy()
    print("Learned policy:", policy)
    
    agent.print_policy_grid()
    
    # Test the policy
    state, _ = env.reset()
    env.render()
    
    total_reward = 0
    terminated = truncated = False
    
    while not (terminated or truncated):
        action = policy.get(state, 0)  # Use default action 0 if state not in policy
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        env.render()
    
    print(f"Episode ended with total reward: {total_reward}")