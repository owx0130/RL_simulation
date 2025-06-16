"""
params.py

This module defines the parameters for the maritime simulation: environment, agent, display, and penalties.
"""

# Add file directory to system path and change working directory (to maintain imports)
import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.resolve()))
os.chdir(str(Path(__file__).parent.parent.resolve()))

from RL.reference_values import *
from RL.helper_funcs import *

# Display (square screen)
screen_height = 800
right_column_width = 300
screen_width = screen_height + right_column_width
margins = 48
grid_number = 8
display_rate = 30  # Hz
entity_size = 8  # of agent, obstacles, goal (metres)
colours_dict = {
    0: ["Unclassified", WHITE],
    1: ["Heading Away", LIME],
    2: ["Head-on", RED],
    3: ["Overtaking", ORANGE],
    4: ["Crossing STBD", CYAN],
    5: ["Crossing PORT", CYAN]
}

# Environment
agent_start_pos_longlat = (103.859400,1.4221174)
goal_pos_longlat = (103.86100,1.4211174) 
heading = 90.0  # 90 degrees = east
decision_rate = 30  # Hz
ops_bubble_multiplier = 0.75
proximity_to_goal = 5  # metres
max_obstacles = 10
simulation_status = True

# Reward/penalty weightages
distance_change_weightage = 1
time_penalty_weightage = -0.1
exceed_ops_env_penalty_weightage = -400
goal_reward_weightage = 300
obs_collision_penalty_weightage = -400
too_close_to_obstacle_penalty_weightage = -2

# COLREGs related reward/penalty weightages
obs_head_on_weightage = 200
obs_overtaking_weightage = 200
obs_crossing_weightage = 200
obs_turning_correctly_weightage = 1

rewards_weights_dict = {
    "distance_change_weightage": distance_change_weightage,
    "time_penalty_weightage": time_penalty_weightage,
    "exceed_ops_env_penalty_weightage": exceed_ops_env_penalty_weightage,
    "goal_reward_weightage": goal_reward_weightage,
    "obs_collision_penalty_weightage": obs_collision_penalty_weightage,
    "too_close_to_obstacle_penalty_weightage": too_close_to_obstacle_penalty_weightage,
    "obs_head_on_weightage": obs_head_on_weightage,
    "obs_crossing_weightage": obs_crossing_weightage,
    "obs_turning_correctly_weightage": obs_turning_correctly_weightage,
    "obs_overtaking_weightage": obs_overtaking_weightage
}

# Agent properties
max_velocity_knots = 33
cruising_speed_knots = max_velocity_knots / 3
max_acc_ms2 = 2  # meters per second squared
max_yaw_rate_degs = 45  # per second
detection_radius = 100  # metres

# Safety Radius (metres)
small_radius = 30
medium_radius = 40
large_radius = 60

safety_radius_dict = {
    "small": small_radius,
    "medium": medium_radius,
    "large": large_radius,
}
