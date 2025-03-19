import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Tuple
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm

class ImprovedMonteCarlo:
    def __init__(self, action_space: int, gamma: float=0.99, initial_epsilon: float=1.0, final_epsilon: float=0.01, 
                 epsilon_decay: float=0.995, optimistic_init: float=0.0):
        self.action_space: int = action_space
        self.gamma: float = gamma
        self.initial_epsilon: float = initial_epsilon
        self.epsilon: float = initial_epsilon
        self.final_epsilon: float = final_epsilon
        self.epsilon_decay: float = epsilon_decay
        
        # Optimistic initialization of Q-values
        self.Q: Dict[Any, np.ndarray] = defaultdict(lambda: np.ones(action_space) * optimistic_init)
        self.N: Dict[Any, np.ndarray] = defaultdict(lambda: np.zeros(action_space))
        
        # For tracking performance
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = []
        self.success_window = 100

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

    def epsilon_greedy(self, state: Any) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space)
        return int(np.argmax(self.Q[state]))

    def generate_episode(self, env: gym.Env) -> List[Tuple[Any, int, float]]:
        episode = []
        state, _ = env.reset()
        terminated = truncated = False
        episode_reward = 0
        episode_length = 0

        while not (terminated or truncated):
            action: int = self.epsilon_greedy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            episode_reward += reward
            episode_length += 1
            
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.success_rate.append(1 if episode_reward > 0 else 0)
        
        return episode

    def update_q_values(self, episode: List[Tuple[Any, int, float]]) -> None:
        G: float = 0
        visited: Dict[Tuple[Any, int], int] = {}  # Track first visit with index
        
        # First, identify the first occurrence of each state-action pair
        for i, (state, action, _) in enumerate(episode):
            if (state, action) not in visited:
                visited[(state, action)] = i
                
        # Now process the episode
        for i in reversed(range(len(episode))):
            state, action, reward = episode[i]
            G = reward + self.gamma * G
            
            # If this is the first visit to (s,a) in this episode
            if visited.get((state, action)) == i:
                self.N[state][action] += 1
                # Use decreasing learning rate for stability
                alpha: float = 1 / self.N[state][action]
                self.Q[state][action] += alpha * (G - self.Q[state][action])

    def train(self, env: gym.Env, episodes: int = 10000, verbose: bool = True) -> None:
        """Train the agent"""
        with tqdm(total=episodes, disable=not verbose) as pbar:
            for _ in range(episodes):
                episode = self.generate_episode(env)
                self.update_q_values(episode)
                self.decay_epsilon()
                
                # Update progress bar with useful info
                if len(self.success_rate) >= self.success_window:
                    recent_success = np.mean(self.success_rate[-self.success_window:])
                else:
                    recent_success = np.mean(self.success_rate) if self.success_rate else 0
                    
                pbar.update(1)
                pbar.set_postfix({
                    'epsilon': f'{self.epsilon:.3f}',
                    'success': f'{recent_success:.2f}',
                    'reward': f'{self.episode_rewards[-1]:.1f}'
                })

    def get_policy(self):
        """Extract the greedy policy from Q-values"""
        policy: Dict[Any, int] = {state: int(np.argmax(actions)) for state, actions in self.Q.items()}
        return policy
    
    def print_policy_grid(self, shape=(4, 4)):
        """Print the policy as a grid"""
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
                    row += "□ "  # Empty square for states without policy
            print(row)
    
    def plot_training_results(self):
        """Plot training performance metrics"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].plot(self.episode_rewards)
        axes[0].set_title('Episode Rewards')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        
        axes[1].plot(self.episode_lengths)
        axes[1].set_title('Episode Lengths')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Steps')
        
        if len(self.success_rate) >= self.success_window:
            moving_avg = [np.mean(self.success_rate[i:i+self.success_window]) 
                          for i in range(len(self.success_rate) - self.success_window + 1)]
            axes[2].plot(moving_avg)
            axes[2].set_title(f'Success Rate (Moving Avg, Window={self.success_window})')
            axes[2].set_xlabel('Episode')
            axes[2].set_ylabel('Success Rate')
            axes[2].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.show()

def test_policy(env, policy, n_episodes=10, render=False):
    """Test the policy on the environment"""
    rewards = []
    lengths = []
    successes = 0
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        if render:
            print("Starting new episode")
            print(env.render())
            
        while not done:
            if state in policy:
                action = policy[state]
            else:
                action = np.random.randint(env.action_space.n)  # Random action if state not in policy
                
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            state = next_state
            
            if render:
                print(f"Action: {action}")
                print(env.render())
                print(f"Reward: {reward}")
                
        rewards.append(total_reward)
        lengths.append(steps)
        if total_reward > 0:
            successes += 1
            
    success_rate = successes / n_episodes
    avg_reward = np.mean(rewards)
    avg_length = np.mean(lengths)
    
    print(f"Success Rate: {success_rate:.2f}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Length: {avg_length:.2f}")
    
    return success_rate, avg_reward, avg_length

if __name__ == "__main__":
    # Create the environment
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
    
    agent = ImprovedMonteCarlo(
        action_space=env.action_space.n,
        gamma=0.99,             # Discount factor
        initial_epsilon=1.0,    # Start with full exploration
        final_epsilon=0.01,     # Minimum exploration rate
        epsilon_decay=0.995,    # Gradually reduce exploration
        optimistic_init=1.0     # Optimistic initialization
    )
    
    agent.train(env=env, episodes=5000)
    
    policy = agent.get_policy()
    print("Learned policy:", policy)
    
    agent.print_policy_grid()
    
    agent.plot_training_results()
    
    print("\nTesting the learned policy:")
    test_policy(env, policy, n_episodes=100, render=False)
    
    print("\nDemonstrating one episode with the learned policy:")
    test_policy(env, policy, n_episodes=1, render=True)