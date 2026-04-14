"""
A/B Testing with Multi-Armed Bandits
======================================
This script implements and compares two popular Multi-Armed Bandit strategies:
Epsilon-Greedy (with 1/t decay) and Thompson Sampling (with Gaussian prior).

The experiment simulates four advertisement options with true mean rewards
of [1, 2, 3, 4]. Results are logged, saved to a CSV, and visualized.

Requirements:
    numpy, matplotlib, loguru

Usage:
    python Bandit.py
"""

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from loguru import logger
import csv
import random

# --- Logger Configuration ---
# INFO level goes to terminal; DEBUG level (detailed trial info) goes to file.
logger.remove()
logger.add(lambda msg: print(msg, end=""), level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
logger.add("bandit_experiment.log", level="DEBUG", format="{time} | {level} | {message}")

############################### ABSTRACT BASE CLASS

class Bandit(ABC):
    """
    Abstract base class for a Multi-Armed Bandit algorithm.
    All methods must be implemented by subclasses to ensure compatibility
    with the experiment runner.
    """

    @abstractmethod
    def __init__(self, p):
        """Initializes the bandit with true reward means."""
        pass

    @abstractmethod
    def __repr__(self):
        """String representation of the bandit algorithm."""
        pass

    @abstractmethod
    def pull(self):
        """Decision logic to select an arm and receive a reward."""
        pass

    @abstractmethod
    def update(self, arm, reward):
        """Updates internal estimates based on the observed reward."""
        pass

    @abstractmethod
    def experiment(self, trials):
        """Runs the simulation for a specified number of iterations."""
        pass

    @abstractmethod
    def report(self):
        """Calculates final metrics and saves data to CSV."""
        pass

############################### VISUALIZATION CLASS

class Visualization:
    """
    Handles the generation and storage of performance plots.
    """

    def plot1(self, eg_history, ts_history):
        """
        Plots the cumulative rewards on Linear and Log scales.

        Args:
            eg_history (list): Rewards history from Epsilon-Greedy.
            ts_history (list): Rewards history from Thompson Sampling.
        """
        eg_cum = np.cumsum(eg_history)
        ts_cum = np.cumsum(ts_history)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(eg_cum, label="E-Greedy")
        axes[0].plot(ts_cum, label="Thompson")
        axes[0].set_title("Learning Process (Linear)")
        axes[0].set_xlabel("Trial")
        axes[0].set_ylabel("Total Reward")
        axes[0].legend()

        axes[1].plot(eg_cum, label="E-Greedy")
        axes[1].plot(ts_cum, label="Thompson")
        axes[1].set_yscale('log')
        axes[1].set_title("Learning Process (Log Scale)")
        axes[1].set_xlabel("Trial")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig("plot1_learning_process.png")
        logger.info("Saved plot1_learning_process.png")
        plt.show()

    def plot2(self, eg_history, ts_history, eg_regret, ts_regret):
        """
        Plots a side-by-side comparison of Cumulative Rewards and Regrets.

        Args:
            eg_history/ts_history: Reward sequences.
            eg_regret/ts_regret: Regret sequences (Optimal Mean - Actual Mean).
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(np.cumsum(eg_history), label="E-Greedy")
        axes[0].plot(np.cumsum(ts_history), label="Thompson")
        axes[0].set_title("Cumulative Rewards")
        axes[0].set_xlabel("Trial")
        axes[0].legend()

        axes[1].plot(np.cumsum(eg_regret), label="E-Greedy")
        axes[1].plot(np.cumsum(ts_regret), label="Thompson")
        axes[1].set_title("Cumulative Regret")
        axes[1].set_xlabel("Trial")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig("plot2_comparison.png")
        logger.info("Saved plot2_comparison.png")
        plt.show()

############################### EPSILON-GREEDY

class EpsilonGreedy(Bandit):
    """
    Implementation of the Epsilon-Greedy algorithm with 1/t decay.
    """
    def __init__(self, p):
        self.p = p  # True means [1, 2, 3, 4]
        self.n_arms = len(p)
        self.counts = [0] * self.n_arms      # Times each arm was pulled
        self.values = [0.0] * self.n_arms    # Running average reward per arm
        self.reward_history = []
        self.regret_history = []
        self.arm_history = []
        self.total_reward = 0

    def __repr__(self):
        return f"EpsilonGreedy(n_arms={self.n_arms})"

    def pull(self):
        """
        Explores with probability 1/t, otherwise exploits the current best arm.
        """
        trial = len(self.reward_history) + 1
        epsilon = 1.0 / trial  # Decay epsilon based on the trial number

        if random.random() < epsilon:
            arm = random.randint(0, self.n_arms - 1) # Random exploration
        else:
            arm = np.argmax(self.values) # Greedy exploitation

        # Draw reward from a Normal distribution (Gaussian noise)
        reward = np.random.normal(self.p[arm], 1.0)
        return arm, reward

    def update(self, arm, reward):
        """Updates the running mean for the selected arm."""
        self.counts[arm] += 1
        # Incremental average formula: Q_t+1 = Q_t + 1/n * [R_t - Q_t]
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

        self.total_reward += reward
        self.reward_history.append(reward)
        # Regret = Best possible mean (4) - Expected mean of chosen arm
        self.regret_history.append(max(self.p) - self.p[arm])
        self.arm_history.append(arm)

    def experiment(self, trials=20000):
        """Runs the simulation for N trials."""
        logger.debug(f"EpsilonGreedy: starting {trials} trials.")
        for _ in range(trials):
            arm, reward = self.pull()
            self.update(arm, reward)

    def report(self):
        """Logs summary stats and appends trial data to bandit_rewards.csv."""
        avg_reward = self.total_reward / len(self.reward_history)
        cum_regret = sum(self.regret_history)

        logger.info(f"[EpsilonGreedy] Average Reward   : {avg_reward:.4f}")
        logger.info(f"[EpsilonGreedy] Cumulative Regret: {cum_regret:.4f}")

        # Append data to the shared CSV file
        with open("bandit_rewards.csv", "a", newline="") as f:
            writer = csv.writer(f)
            for i in range(len(self.reward_history)):
                writer.writerow([self.arm_history[i], self.reward_history[i], "EpsilonGreedy"])

############################### THOMPSON SAMPLING

class ThompsonSampling(Bandit):
    """
    Thompson Sampling using a Gaussian prior with known precision.
    """
    def __init__(self, p, precision=1.0):
        self.p = p
        self.n_arms = len(p)
        self.precision = precision
        self.means = [0.0] * self.n_arms      # Posterior means
        self.precisions = [precision] * self.n_arms # Posterior precisions
        self.reward_history = []
        self.regret_history = []
        self.arm_history = []
        self.total_reward = 0

    def __repr__(self):
        return f"ThompsonSampling(n_arms={self.n_arms})"

    def pull(self):
        """
        Samples from each arm's posterior distribution and selects the maximum.
        """
        # Sample theta_i ~ N(mean_i, 1/precision_i)
        samples = [np.random.normal(self.means[i], 1.0 / np.sqrt(self.precisions[i]))
                   for i in range(self.n_arms)]
        arm = np.argmax(samples)

        # Receive the actual reward
        reward = np.random.normal(self.p[arm], 1.0)
        return arm, reward

    def update(self, arm, reward):
        """Performs a Bayesian update on the Normal distribution parameters."""
        old_precision = self.precisions[arm]
        # Update precision and mean (Conjugate prior for Normal with known variance)
        self.precisions[arm] += 1.0
        self.means[arm] = (old_precision * self.means[arm] + reward) / self.precisions[arm]

        self.total_reward += reward
        self.reward_history.append(reward)
        self.regret_history.append(max(self.p) - self.p[arm])
        self.arm_history.append(arm)

    def experiment(self, trials=20000):
        """Runs the Thompson Sampling simulation."""
        logger.debug(f"ThompsonSampling: starting {trials} trials.")
        for _ in range(trials):
            arm, reward = self.pull()
            self.update(arm, reward)

    def report(self):
        """Logs results and appends data to bandit_rewards.csv."""
        avg_reward = self.total_reward / len(self.reward_history)
        cum_regret = sum(self.regret_history)

        logger.info(f"[ThompsonSampling] Average Reward   : {avg_reward:.4f}")
        logger.info(f"[ThompsonSampling] Cumulative Regret: {cum_regret:.4f}")

        with open("bandit_rewards.csv", "a", newline="") as f:
            writer = csv.writer(f)
            for i in range(len(self.reward_history)):
                writer.writerow([self.arm_history[i], self.reward_history[i], "ThompsonSampling"])

############################### MAIN COMPARISON

def comparison():
    """
    Initializes bandits, runs experiments, and executes visualizations.
    Ensures that the CSV file is created with headers first.
    """
    BANDIT_REWARDS = [1, 2, 3, 4]
    TRIALS = 20000

    # 1. Prepare CSV file {Bandit, Reward, Algorithm}
    with open("bandit_rewards.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Bandit", "Reward", "Algorithm"])

    # 2. Run Epsilon-Greedy Experiment
    logger.info("=" * 55)
    logger.info("Starting the EpsilonGreedy experiment.")
    eg = EpsilonGreedy(BANDIT_REWARDS)
    eg.experiment(TRIALS)
    eg.report()

    # 3. Run Thompson Sampling Experiment
    logger.info("=" * 55)
    logger.info("Starting the ThompsonSampling experiment.")
    ts = ThompsonSampling(BANDIT_REWARDS)
    ts.experiment(TRIALS)
    ts.report()

    # 4. Compare Visually
    logger.info("=" * 55)
    logger.info("Comparing the results of both algorithms.")
    viz = Visualization()
    viz.plot1(eg.reward_history, ts.reward_history)
    viz.plot2(eg.reward_history, ts.reward_history, eg.regret_history, ts.regret_history)
    logger.info("All done.")

if __name__ == '__main__':
    comparison()