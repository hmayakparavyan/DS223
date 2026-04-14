# Multi-Armed Bandit: A/B Testing Experiment

This project implements and compares **Epsilon-Greedy** and **Thompson Sampling** strategies for a 4-arm bandit problem. 

## Experiment Design
* **Bandit Rewards**: [1, 2, 3, 4].
* **Number of Trials**: 20,000.
* **Epsilon-Greedy**: Implements $1/t$ decay.
* **Thompson Sampling**: Designed with known precision.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`.
2. Run the simulation: `python Bandit.py`.

## Outputs
* **CSV Data**: Rewards are stored in `bandit_rewards.csv` with columns `{Bandit, Reward, Algorithm}`.
* **Visualizations**: 
    * `plot1()`: Learning process (linear and log).
    * `plot2()`: Cumulative rewards and regrets.

## 💡 Bonus Plan
My suggested implementation plan for better performance includes:
1. **Bayesian Regret Analysis**: Calculating expected regret for smoother convergence curves.
2. **Hyperparameter Warm-up**: Implementing a small initial exploration phase to stabilize early reward estimates.