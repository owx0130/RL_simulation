"""
visualize_model.py

This script loads a trained Proximal Policy Optimization (PPO) agent and uses it to interact with and render a custom environment defined in `env.py`.

The script creates an instance of the PPO agent, loads the saved model parameters, and runs the agent in the environment, rendering each step.

Usage:
    Run this script to visualize the performance of the trained PPO agent.

Example:
    $ python visualize_model.py
"""

from stable_baselines3 import PPO
from env import MyEnv  # Import your custom environment module
from params import *

# Create the environment
env = MyEnv(
        agent_start_pos_longlat,
        goal_pos_longlat,
        heading,
        max_velocity_knots,
        cruising_speed_knots,
        max_acc_ms2,
        max_yaw_rate_degs,
        detection_radius,
        min_obs_detection_radius,
        screen_height,
        screen_width,
        margins,
        ops_bubble_multiplier,
        grid_number,
        decision_rate,
        display_rate,
        colours_dict,
        max_obstacles,
        safety_radius_dict,
        rewards_weights_dict,
        entity_size,
        proximity_to_goal,
        
        obstacle_motion_type,   
        max_spawned_obs=no_of_generated_obs,
        
        simulation_status=True,
        record=False,
        video_name="model_performance",
    )

# Load the trained model
model = PPO.load(r"RL_training\simple_nav\models\best.zip")

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
