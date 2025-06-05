"""
visualize_model.py

This script loads a trained Proximal Policy Optimization (PPO) agent and uses it to interact with and render a custom environment defined in `env.py`.

The script creates an instance of the PPO agent, loads the saved model parameters, and runs the agent in the environment, rendering each step.

Usage:
    Run this script to visualize the performance of the trained PPO agent.

Example:
    $ python visualize_model.py
"""

from stable_baselines3 import PPO, SAC
from params import *
from model_funcs import *

# Create the environment
env = create_env(difficulty=4, record=True)

# Load the trained model
model = SAC.load("RL_training/sac_model.zip")

# Reset the environment and get the initial observation
obs, _ = env.reset()

# Visualize the model
terminated = False
truncated = False
total_reward = 0
while not terminated and not truncated:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
    total_reward += reward
print("Total Rewards: ", total_reward)
env.close()
