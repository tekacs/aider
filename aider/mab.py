import json
import math
import os
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import numpy as np


class MultiArmedBandit(ABC):
    """Base class for Multi-Armed Bandit algorithms."""

    def __init__(self, arms, state_file=None):
        self.arms = list(arms)  # List of model names (arms)
        self.state_file = Path(state_file) if state_file else None
        self.counts = defaultdict(int)  # Number of times each arm was pulled
        self.rewards = defaultdict(float)  # Sum of rewards for each arm
        self.load_state()

    @abstractmethod
    def select_arm(self):
        """Select an arm (model) to pull."""
        pass

    def update(self, arm, reward):
        """Update the state after pulling an arm and receiving a reward."""
        if arm not in self.arms:
            # Dynamically add new arms if encountered
            self.arms.append(arm)
        self.counts[arm] += 1
        self.rewards[arm] += reward
        self.save_state()

    def get_average_reward(self, arm):
        """Calculate the average reward for a given arm."""
        if self.counts[arm] == 0:
            return 0.0
        return self.rewards[arm] / self.counts[arm]

    def save_state(self):
        """Save the current state (counts and rewards) to a file."""
        if not self.state_file:
            return
        try:
            state = {
                "arms": self.arms,
                "counts": dict(self.counts),
                "rewards": dict(self.rewards),
            }
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=4)
        except OSError as e:
            print(f"Warning: Could not save MAB state to {self.state_file}: {e}")

    def load_state(self):
        """Load the state from a file."""
        if not self.state_file or not self.state_file.exists():
            return
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
            # Update arms, ensuring loaded arms are included
            loaded_arms = set(state.get("arms", []))
            self.arms = list(set(self.arms) | loaded_arms)
            self.counts = defaultdict(int, state.get("counts", {}))
            self.rewards = defaultdict(float, state.get("rewards", {}))
        except (OSError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load MAB state from {self.state_file}: {e}")


class EpsilonGreedy(MultiArmedBandit):
    """Epsilon-Greedy MAB algorithm."""

    def __init__(self, arms, state_file=None, epsilon=0.1):
        super().__init__(arms, state_file)
        self.epsilon = epsilon

    def select_arm(self):
        if random.random() > self.epsilon:
            # Exploit: choose the best arm based on average reward
            best_arm = max(self.arms, key=self.get_average_reward)
            # Handle ties by random choice among best arms
            max_reward = self.get_average_reward(best_arm)
            best_arms = [
                arm for arm in self.arms if self.get_average_reward(arm) == max_reward
            ]
            return random.choice(best_arms)
        else:
            # Explore: choose a random arm
            return random.choice(self.arms)


class UCB1(MultiArmedBandit):
    """Upper Confidence Bound (UCB1) MAB algorithm."""

    def select_arm(self):
        total_pulls = sum(self.counts.values())

        # Ensure all arms are pulled at least once initially
        for arm in self.arms:
            if self.counts[arm] == 0:
                return arm

        ucb_values = {}
        for arm in self.arms:
            average_reward = self.get_average_reward(arm)
            exploration_term = math.sqrt(
                (2 * math.log(total_pulls)) / self.counts[arm]
            )
            ucb_values[arm] = average_reward + exploration_term

        # Choose the arm with the highest UCB value
        best_arm = max(ucb_values, key=ucb_values.get)
        # Handle ties by random choice among best arms
        max_ucb = ucb_values[best_arm]
        best_arms = [arm for arm, ucb in ucb_values.items() if ucb == max_ucb]
        return random.choice(best_arms)


class ThompsonSampling(MultiArmedBandit):
    """Thompson Sampling MAB algorithm (using Beta distribution for binary rewards)."""
    # Assumes rewards are binary (0 or 1) or scaled to [0, 1] for Beta distribution.
    # We'll use alpha = successes + 1, beta = failures + 1

    def __init__(self, arms, state_file=None):
        super().__init__(arms, state_file)
        # Thompson Sampling specific state: alpha and beta parameters for Beta distribution
        self.alpha = defaultdict(lambda: 1.0) # Corresponds to successes + 1
        self.beta = defaultdict(lambda: 1.0)  # Corresponds to failures + 1
        self._initialize_alpha_beta()

    def _initialize_alpha_beta(self):
        """Initialize alpha/beta based on loaded counts and rewards."""
        for arm in self.arms:
            successes = self.rewards[arm] # Assuming reward is success count (0 or 1)
            failures = self.counts[arm] - successes
            self.alpha[arm] = successes + 1.0
            self.beta[arm] = failures + 1.0

    def select_arm(self):
        sampled_theta = {}
        for arm in self.arms:
            # Sample from the Beta distribution for each arm
            sampled_theta[arm] = np.random.beta(self.alpha[arm], self.beta[arm])

        # Choose the arm with the highest sampled value
        best_arm = max(sampled_theta, key=sampled_theta.get)
        return best_arm

    def update(self, arm, reward):
        """Update alpha and beta parameters based on the reward (assumed 0 or 1)."""
        super().update(arm, reward) # Update counts and total rewards

        # Update alpha/beta. Reward=1 increases alpha, Reward=0 increases beta.
        # Clamp reward to [0, 1] for Beta distribution interpretation
        reward = max(0.0, min(1.0, reward))
        if reward >= 0.5: # Treat >= 0.5 as success
             self.alpha[arm] += 1.0
        else: # Treat < 0.5 as failure
             self.beta[arm] += 1.0

        # Re-save state including alpha/beta if needed (optional, base class saves counts/rewards)
        # self.save_state() # Base class already calls this

    def save_state(self):
        """Save the current state including alpha and beta."""
        if not self.state_file:
            return
        try:
            state = {
                "arms": self.arms,
                "counts": dict(self.counts),
                "rewards": dict(self.rewards),
                "alpha": dict(self.alpha),
                "beta": dict(self.beta),
            }
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=4)
        except OSError as e:
            print(f"Warning: Could not save MAB state to {self.state_file}: {e}")

    def load_state(self):
        """Load the state including alpha and beta."""
        if not self.state_file or not self.state_file.exists():
            return
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
            # Update arms, ensuring loaded arms are included
            loaded_arms = set(state.get("arms", []))
            self.arms = list(set(self.arms) | loaded_arms)
            self.counts = defaultdict(int, state.get("counts", {}))
            self.rewards = defaultdict(float, state.get("rewards", {}))
            self.alpha = defaultdict(lambda: 1.0, state.get("alpha", {}))
            self.beta = defaultdict(lambda: 1.0, state.get("beta", {}))
            # Ensure consistency after loading
            self._initialize_alpha_beta()
        except (OSError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load MAB state from {self.state_file}: {e}")


def get_mab_instance(algorithm_name, arms, state_file):
    """Factory function to get a MAB instance based on name."""
    if algorithm_name == "epsilon-greedy":
        return EpsilonGreedy(arms, state_file)
    elif algorithm_name == "ucb1":
        return UCB1(arms, state_file)
    elif algorithm_name == "thompson":
        return ThompsonSampling(arms, state_file)
    else:
        raise ValueError(f"Unknown MAB algorithm: {algorithm_name}")
