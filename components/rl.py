"""
Reinforcement Learning components.
"""


def lr_demo():
    """
    Reinforcement Learning demonstration: Multi-armed bandit with epsilon-greedy.
    """
    import numpy as np

    # Reward function or Environment
    class Bandit:
        """A simple k-armed bandit with Bernoulli rewards."""

        def __init__(self, k=10):
            self.k = k
            # True reward probabilities for each arm (unknown to the agent)
            self.q_true = np.random.rand(k)  # Uniform between 0 and 1

        def pull(self, arm):
            """Return reward (0 or 1) for pulling the given arm."""
            return np.random.rand() < self.q_true[arm]

    # Agent or Actor
    class EpsilonGreedyAgent:
        """Epsilon-greedy agent for the bandit problem."""

        def __init__(self, k, epsilon=0.1):
            self.k = k
            self.epsilon = epsilon
            self.q_estimates = np.zeros(k)  # Estimated values for each arm
            self.action_counts = np.zeros(k)  # Number of times each arm was pulled

        # policy function
        def select_action(self):
            """Select an action using epsilon-greedy policy."""
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.k)  # Explore
            else:
                return np.argmax(self.q_estimates)  # Exploit

        def update(self, action, reward):
            """Update the action-value estimate for the taken action."""
            self.action_counts[action] += 1
            # Incremental update rule: Q_{n+1} = Q_n + (1/n)[R_n - Q_n]
            self.q_estimates[action] += (
                reward - self.q_estimates[action]
            ) / self.action_counts[action]

    def train_bandit(agent, bandit, steps=1000):
        """Train the agent on the bandit for a given number of steps."""
        rewards = np.zeros(steps)
        for step in range(steps):
            action = agent.select_action()
            reward = bandit.pull(action)
            agent.update(action, reward)
            rewards[step] = reward
        return rewards

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create a bandit with 10 arms
    k_arms = 10
    bandit = Bandit(k=k_arms)

    # Create an agent with epsilon=0.1
    agent = EpsilonGreedyAgent(k=k_arms, epsilon=0.1)

    # Train the agent
    print("Training agent on a 10-armed bandit with Bernoulli rewards...")
    rewards = train_bandit(agent, bandit, steps=2000)

    # Print results
    print(f"Average reward: {np.mean(rewards):.3f}")
    print(f"Estimated values: {agent.q_estimates}")
    print(f"True values:      {bandit.q_true}")
    print(f"Optimal arm:      {np.argmax(bandit.q_true)}")
    print(f"Agent's best arm: {np.argmax(agent.q_estimates)}")

    # Show how often the agent picked the optimal arm
    optimal_arm = np.argmax(bandit.q_true)
    # We don't have the history of actions, so we'll rerun and track
    agent = EpsilonGreedyAgent(k=k_arms, epsilon=0.1)
    optimal_actions = 0
    for _ in range(2000):
        action = agent.select_action()
        reward = bandit.pull(action)
        agent.update(action, reward)
        if action == optimal_arm:
            optimal_actions += 1
    print(f"Optimal arm selection rate: {optimal_actions / 2000:.3f}")


lr_demo()
