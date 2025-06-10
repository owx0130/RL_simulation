"""
env.py

This module defines the MyEnv class, which represents the environment for the maritime simulation.
It includes the initialization of the environment, agent properties, and methods for resetting and stepping through the environment.
Maritime Agent Reinforcement Learning for Intelligent Navigation (MARLIN)

Classes:
    MyEnv: Custom environment for maritime simulation, inheriting from gym.Env.

Main Methods:
    __init__(self, etc):
        Initializes the environment with the given parameters.
    reset(self):
        Resets the environment to its initial state.
    step(self, action):
        Executes a step in the environment based on the given action.
    render(self):
        Renders the environment with pygame.

Main attributes:

    self.action_space: The action space for the environment. 
        Bounds:
            Normalised throttle output: [0, 1]
            Normalised nozzle angle: [-1,1]

    self.observation_space: The observation space for the environment. 
        Format:
        {'agent': distance_to_goal[0, 1], angle_to_goal[-1, 1], velocity[0,1], heading[0,1]] 
        'obs1_active': [0,1], 'obs1_type': [0,1,2,3,4], 'obs1': [distance_to_agent[0,1], angle_to_agent[-1,1], velocity[0,1], heading[0,1], safety_radius]
        'obs2_active': [0,1], 'obs2_type': [0,1,2,3,4], 'obs2': [distance_to_agent[0,1], angle_to_agent[-1,1], velocity[0,1], heading[0,1], safety_radius] 
        ...}
        
        where:
            obs_active:
                • 0 => inactive 
                • 1 => active
            obs_type:
                • 0 => unclassified
                • 1 => heading away
                • 2 => head on
                • 3 => crossing
                • 4 => overtaking
    
    self.state: The current state of the environment. Follows self.observation_space format.
    
    self.prev_state: The previous state of the environment. Mainly for rendering purposes. Follows self.observation_space format.

Usage:
    Import this module and create an instance of MyEnv.

Example:
    from env import MyEnv
"""

# Add file directory to system path and change working directory (to maintain imports)
import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.resolve()))
os.chdir(str(Path(__file__).parent.parent.resolve()))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import copy
import scipy.optimize as opt
import cv2
from RL.helper_funcs import *
from RL.reference_values import *

from collections import deque

# Object to represent the agent
class Agent():

    def __init__(
        self,
        start_pos: np.ndarray,  # xy, metres
        start_heading: float,  # degrees °
        start_velocity: float,  # m/s
        max_velocity: float  # m/s
    ):
        self.xy = start_pos  # agent xy position in metres
        self.velocity = start_velocity  # agent velocity in m/s
        self.heading = start_heading  # agent heading in degrees
        self.max_velocity_ms = max_velocity  # max agent velocity

    def update(self, acceleration: float, yaw_rate: float, time_step: float):
        """Update state of agent with acceleration and yaw rate values"""
        
        heading_rad = np.radians(self.heading)
        self.xy += np.array([np.sin(heading_rad), np.cos(heading_rad)]) * self.velocity * time_step
        
        self.velocity += acceleration * time_step
        self.heading += yaw_rate * time_step
        self.heading %= 360

        self.velocity = np.clip(self.velocity, 0, self.max_velocity_ms) 
        
        # Simulate drag to agent's linear motion *(not accurate representation of actual drag)
        if acceleration == 0 and self.velocity > 0:
            drag_coefficient = 0.1
            wetted_area = 2 * 0.8  # metre ^2
            water_density = 1000  # kg/m^3
            mass = 40  # kg
            
            drag_force = 0.5 * water_density * drag_coefficient * wetted_area * (self.velocity ** 2)
            deceleration = drag_force / mass
            
            self.velocity = max(self.velocity - deceleration * time_step, 0)
            if self.velocity <= 0.1:
                self.velocity = 0

        return self.xy, self.velocity, self.heading 

# Object to represent the physical obstacles
class Obstacle():

    def __init__(
        self,
        start_pos: np.ndarray,  # xy, metres
        start_heading: float,  # degrees °
        start_velocity: float,  # m/s
        safety_radius: int,  # m
        isActive: bool,  # [0,1]
        initial_heading_diff
    ):
        self.xy = start_pos  # agent xy position in metres
        self.velocity = start_velocity  # agent velocity in m/s
        self.heading = start_heading  # agent heading in degrees
        self.safety_radius = safety_radius
        self.isActive = isActive
        self.type = 0
        self.isRewardGiven = False
        self.sin_initial_heading_diff = np.sin(np.radians(initial_heading_diff))
        self.cos_initial_heading_diff = np.cos(np.radians(initial_heading_diff))

    def update(self, time_step: float):

        heading_rad = np.radians(self.heading)
        self.xy += np.array([np.sin(heading_rad), np.cos(heading_rad)]) * self.velocity * time_step  # Update position

class MyEnv(gym.Env):
    def __init__(
        self,

        # Ops environment related attributes
        agent_start_pos_longlat,
        goal_pos_longlat,
        heading_deg,
        decision_rate,
        ops_bubble_multiplier,
        proximity_to_goal,
        max_obstacles,
        safety_radius_dict,
        rewards_weights_dict,

        # Display related attributes
        screen_height,
        screen_width,
        margins,
        grid_number,
        display_rate,
        entity_size,
        colours_dict,

        # Agent related attributes
        max_velocity_knots,
        cruising_speed_knots,
        max_acc_ms2,
        max_yaw_rate_degs,
        detection_radius,

        record,
        difficulty,
        simulation_status = True,
        video_name = "RL_training/Current"
    ):
        super(MyEnv, self).__init__()
        
        # Agent properties
        self.max_velocity_ms = knotstoms(max_velocity_knots)
        self.cruising_speed_ms = knotstoms(cruising_speed_knots)
        self.max_acc_ms2 = max_acc_ms2
        self.max_yaw_rate_degs = max_yaw_rate_degs
        self.detection_radius = detection_radius

        # Navigational data
        self.goal_pos_xy = None
        self.agent_start_pos_xy = None
        self.agent_start_pos_xy_rel = None  # relative position to goal pos
        self.agent_angle_to_goal = None
        self.agent_dist_to_goal = None
        self.initial_heading_degs = None
        
        # Operation Environment for simulation
        self.ops_bubble_multiplier = ops_bubble_multiplier
        self.ops_COG = None
        self.ops_bubble_radius = None
        self.ops_bottom_left = None
        self.ops_top_right = None
        self.max_ops_dist = None
        self.max_dist_in_boundary = None  # diameter of circular boundary
        self.decision_rate = decision_rate
        self.safety_radius_dict = safety_radius_dict 
        self.reward_weights_dict = rewards_weights_dict 
        self.proximity_to_goal = proximity_to_goal
        self.max_obstacles = max_obstacles
        self.max_spawned_obs = 0
        self.min_obs_velocity_ms = self.cruising_speed_ms * 0.5
        self.max_obs_velocity_ms = self.cruising_speed_ms * 1.0
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()
        self.state = None
        self.prev_state = None
        self.time_step = 1 / self.decision_rate
        self.end_time = None
        self.elapsed_time = 0
        self.rewards_log = self.prev_rewards_log = {}
        self.collision_flags = [False] * self.max_obstacles
        self.collided = False
        self.obstacle_motion_type = 0
        self.simulation = simulation_status
        
        # Screen properties
        self.screen = None  # initialized later in render()
        self.screen_height = screen_height  # pixels
        self.screen_width = screen_width  # pixels
        self.left_column_width = self.screen_height  # pixels
        self.margins = margins  # pixels
        self.grid_number = grid_number + 1  # number of grids
        self.display_rate = display_rate
        self.steps = int(self.display_rate / self.decision_rate)
        self.entity_size = entity_size
        self.size_pixels = None  # size of all squares representing agent/obstacles/goal
        self.linewidth_pixels = self.left_column_width // 400  # width of squares
        self.grid_size = self.left_column_width // self.grid_number  # pixel size
        self.grid_scale = None  # metres
        self.colours_dict = colours_dict
        self.agent_pixel_pos_deque = deque([])  # track previous agent pos for drawing the trail line
        self.closest_scale = None
        self.line_length_pixels = None
        
        # Define operating area on screen
        self.drawing_area = pygame.Rect(
            self.margins,
            self.margins,
            self.left_column_width - 2 * self.margins,
            self.left_column_width - 2 * self.margins
        )  # left, top, width, height

        # Record video
        self.video_name = video_name
        self.record = record
        if self.record:
            self.vid_holder = cv2.VideoWriter(
                f"{self.video_name}.mp4",
                cv2.VideoWriter_fourcc(*'mp4v'),
                display_rate,
                (self.screen_width, self.screen_height)
            )

        # Maps environment obstacle id to tracker id from the YOLO object detection
        self.obs_to_tracker_id_dict = {}
        for i in range(1, self.max_obstacles+1):
            self.obs_to_tracker_id_dict[i] = -1
            
        # Difficulty attribute for curriculum learning
        self.difficulty = difficulty
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

    def set_difficulty(self, level):
        self.difficulty = level

    def generate_random_coords(self, lower_bound, upper_bound):
        values = np.concatenate((
            np.arange(-upper_bound, -lower_bound + 1, dtype=np.float32),
            np.arange(lower_bound, upper_bound + 1, dtype=np.float32)
        ))
        return np.random.choice(values, size=2, replace=False)

    def get_action_space(self):
        "Returns initialized action space."
        
        return spaces.Box(low=np.array([0, -1]), high=np.array([1, 1]), dtype=np.float32)

    def get_observation_space(self):
        "Returns initialized observation space"

        # Agent observation space
        # dist_to_goal, sin(angle_diff_to_goal), cos(angle_diff_to_goal), velocity
        observation_space_dict = {
            "agent": spaces.Box(
                low=np.array([0, -1, -1, 0]),
                high=np.array([1, 1, 1, 1]),
                dtype=np.float32,
            )
        }

        # Track whether obstacles are active or not
        observation_space_dict["is_active"] = spaces.MultiBinary(self.max_obstacles)

        # Obstacle observation space
        # dist_to_agent, sin(angle_diff_to_agent), cos(angle_diff_to_agent), velocity, sin(heading_diff), cos(heading_diff)
        observation_space_dict["obstacles"] = spaces.Box(
            low=np.tile(np.array([0, -1, -1, 0, -1, -1, -1, -1]), (self.max_obstacles, 1)),
            high=np.tile(np.array([1] * 8), (self.max_obstacles, 1)),
            shape=(self.max_obstacles, 8),
            dtype=np.float32
        )
        
        # # Help agent identify if obstacle is crossing/overtaking
        # # 0 - heading away/unclassified, 1 - head-on, 2 - overtaking, 3 - crossing stbd, 4 - crossing port
        # observation_space_dict["obstacle_type"] = spaces.MultiBinary([self.max_obstacles, 5])
        
        return spaces.Dict(observation_space_dict)

    def get_signed_angle_diff(self, pt1, heading, pt2):
        """Returns signed angle difference from pt1 to pt2.
        
           CW => -ve
           ACW => +ve
        """
        goal_heading = (90 - np.degrees(np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))) % 360
        angle_diff = (goal_heading - heading + 180) % 360 - 180

        return angle_diff

    def get_operational_environment(self):
        "Returns midpoint, ops_bubble_radius of operational environment"

        # midpoint coordinates
        midpoint = (self.agent_start_pos_xy + self.goal_pos_xy) / 2
        
        # operational radius for training where the agent cannot exceed
        ops_bubble_radius = np.linalg.norm(self.agent_start_pos_xy_rel) * self.ops_bubble_multiplier
        
        # Edges of the map
        min_xy = midpoint - ops_bubble_radius
        max_xy = midpoint + ops_bubble_radius
        
        max_x_dist = max(max_xy[0] - self.goal_pos_xy[0], self.goal_pos_xy[0] - min_xy[0])
        max_y_dist = max(max_xy[1] - self.goal_pos_xy[1], self.goal_pos_xy[1] - min_xy[1])
        
        max_ops_distance = np.array([max_x_dist, max_y_dist])
        
        return midpoint, ops_bubble_radius, min_xy, max_xy, max_ops_distance

    def check_in_operational_environment(self, pos_xy):
        "Checks if a pos_xy point is in the operational environment"
        # distance of point from centre of ops environment
        dist_to_centre = np.linalg.norm(pos_xy - self.ops_COG)
        
        if dist_to_centre < self.ops_bubble_radius: 
            return True
        else: 
            return False

    def closest_distance_with_agent(self, obstacle: Obstacle):
        """Returns the closest projected distance between the 
        agent and an obstacle from the current time until end of episode

        obstacle = Obstacle()"""

        agent_heading_rad = np.radians(self.agent.heading)
        obs_heading_rad = np.radians(obstacle.heading)

        agent_dir = self.agent.velocity * np.array([
            np.sin(agent_heading_rad),
            np.cos(agent_heading_rad)
        ])
        obs_dir = obstacle.velocity * np.array([
            np.sin(obs_heading_rad),
            np.cos(obs_heading_rad)
        ])
        
        def distance(t):
            dist = (self.agent.xy + agent_dir * t) - (obstacle.xy + obs_dir * t)

            return np.dot(dist, dist)

        result = opt.minimize_scalar(
            distance,
            bounds=(0, max(0.01, self.end_time - self.elapsed_time)),
            method="bounded"
        )

        return np.sqrt(result.fun), result.x  # closest distance, time to closest distance

    def classify_obstacle(self, obs: Obstacle):
        "Classifies each obstacle based on its relative position to the agent, velocity and heading"
        
        # Difference in heading between agent and obstacle
        # +ve is clockwise from agent, -ve is anticlockwise from agent
        heading_diff = (obs.heading - self.agent.heading + 180) % 360 - 180
        
        # Relative bearing from agent to obstacle
        vec_to_obs = obs.xy - self.agent.xy
        bearing_to_obs = (90 - np.degrees(np.arctan2(vec_to_obs[1], vec_to_obs[0]))) % 360
        relative_bearing = (bearing_to_obs - self.agent.heading) % 360

        # While training overtaking/head-on/crossing, ensure that obstacle is not reclassified
        # for effective training
        # if self.difficulty == 2 and obs.type != 0 and not 112.5 <= relative_bearing <= 247.5:
        #     return obs.type
        if self.difficulty in [2, 3, 4] and obs.type != 0:
            return obs.type

        # Calculate distance at CPA
        closest_dist, t = self.closest_distance_with_agent(obs)

        # Classify obstacle type accordingly
        if closest_dist < 2 * obs.safety_radius:
            if (160 <= abs(heading_diff) <= 180) and \
               (not 90 <= relative_bearing <= 270):
                # Head on situation
                obs_type = 2
            elif (0 <= abs(heading_diff) <= 20) and \
                 (relative_bearing >= 247.5 or relative_bearing <= 112.5) and \
                 (self.agent.velocity > obs.velocity):
                # Overtaking situation
                obs_type = 3
            elif (relative_bearing <= 90 and -150 <= heading_diff <= -90) or \
                 (relative_bearing <= 112.5 and -90 <= heading_diff <= 0):
                # Crossing on starboard situation
                obs_type = 4
            elif (relative_bearing >= 247.5 and 0 <= heading_diff <= 90) or \
                 (relative_bearing >= 270 and 90 <= heading_diff <= 150):
                # Crossing on port situation
                obs_type = 5
            else:
                obs_type = 1
        else:
            # Obstacle classified as heading away if CPA not close enough
            obs_type = 1
        
        return obs_type

    def get_list_of_collided_obstacles(self):
        collided_obs = []
        for i in range(len(self.obs_list)):
            for j in range(i + 1, len(self.obs_list)):
                obs1 = self.obs_list[i]
                obs2 = self.obs_list[j]

                # Only calculate collision if both obstacles are active
                if obs1.isActive and obs2.isActive:
                    larger_radius = max(obs1.safety_radius, obs2.safety_radius)
                    distance = np.linalg.norm(obs1.xy - obs2.xy)
                    if distance < 0.2 * larger_radius:
                        collided_obs.append(i)
                        collided_obs.append(j)

        return collided_obs

    def check_if_obstacles_too_close(self, obs_xy, obs_safety_radius):
        isTooClose = False
        for existing_obs in self.obs_list:
            # Only compare with active obstacles
            if existing_obs.isActive:
                larger_radius = max(existing_obs.safety_radius, obs_safety_radius)
                if np.linalg.norm(obs_xy - existing_obs.xy) < larger_radius:
                    isTooClose = True
                    break
        
        return isTooClose

    def check_if_too_close_to_goal(self, obs_xy):
        threshold = 30
        if np.linalg.norm(obs_xy - self.goal_pos_xy) < threshold:
            return True
        else:
            return False

    def generate_static_obstacle(self):

        obs_heading = obs_velocity = initial_heading_diff = 0

        # Randomly select obstacle safety radius (select smaller vessels for earlier difficulties)
        # if self.difficulty == 1:
        #     obs_safety_radius = 30
        # else:
        obs_safety_radius = np.random.choice(list(self.safety_radius_dict.values()))

        # Randomly select obstacle position until it is within the ops env and not too close to other obstacles
        # Only done for a max of 500 attempts to prevent infinite loops
        max_attempts = 1000
        for _ in range(max_attempts):
            min_spawn_radius = self.ops_bubble_radius * 0.1
            max_spawn_radius = self.ops_bubble_radius * 0.5
            spawn_radius = np.random.uniform(min_spawn_radius, max_spawn_radius)

            rel_heading = np.random.uniform(0, 360)
            
            # Calculate obstacle spawn position from center of environment
            obs_xy = self.ops_COG + spawn_radius * np.array([
                np.sin(np.radians(rel_heading)),
                np.cos(np.radians(rel_heading))
            ])
            
            # Check if position is within the bounds of the ops area
            isInBounds = self.check_in_operational_environment(obs_xy)
            
            # Check that spawned obstacle is not too close to other obstacles
            isTooClose = self.check_if_obstacles_too_close(obs_xy, obs_safety_radius)
            
            # Check that spawned obstacle is not too close to the goal
            isNearGoal = self.check_if_too_close_to_goal(obs_xy)

            if isInBounds and not isTooClose and not isNearGoal:
                break
        
        # Generate inactive obstacle if impossible to generate within the environment constraints
        if not isInBounds or isTooClose or isNearGoal:
            isActive = False
        else:
            isActive = True
        
        return Obstacle(
            start_pos=obs_xy,
            start_heading=obs_heading,
            start_velocity=obs_velocity,
            safety_radius=obs_safety_radius,
            isActive=isActive,
            initial_heading_diff=initial_heading_diff
        )

    def generate_moving_obstacle(self):
        
        # Randomly select obstacle velocity
        obs_velocity = np.random.uniform(self.min_obs_velocity_ms, self.max_obs_velocity_ms)

        # Randomly select obstacle safety radius
        obs_safety_radius = np.random.choice(list(self.safety_radius_dict.values()))

        # Randomly select obstacle position until it is not too close to other obstacles
        # Only done for a max of 1000 attempts to prevent infinite loops
        max_attempts = 1000
        for _ in range(max_attempts):
            if self.difficulty == 2:
                obs_type = 2
            elif self.difficulty == 3:
                obs_type = 4
            elif self.difficulty == 4:
                obs_type = 3
            else:
                obs_type = np.random.choice([1, 2, 3, 4])

            # Calculate collision point
            if self.agent.velocity == 0:
                time_to_collision = self.end_time - self.elapsed_time
            else:
                time_to_rch_goal = self.agent_dist_to_goal / self.agent.velocity
                time_upper_bound = time_to_rch_goal * 0.7
                time_lower_bound = time_to_rch_goal * 0.3
                
                if time_upper_bound < 3.0:  # Handle case where agent is too close to goal
                    time_to_collision = 3.0
                else:
                    time_to_collision = np.random.uniform(time_lower_bound, time_upper_bound)
                    
            collision_xy = self.agent.xy + self.agent.velocity * time_to_collision * np.array([
                np.sin(np.radians(self.agent.heading)),
                np.cos(np.radians(self.agent.heading))
            ])

            # type 1 will subsequently offset starting position to avoid obstacle
            if obs_type == 1:  # heading away
                rel_heading_to_collision_pt = random_sample((10, 170), (190, 350))
            elif obs_type == 2:  # head on
                rel_heading_to_collision_pt = np.random.uniform(-10, 10)
            elif obs_type == 3:  # overtaking
                rel_heading_to_collision_pt = np.random.uniform(170, 190)
            elif obs_type == 4:  # crossing
                if self.difficulty == 3:
                    # Specifically generate obstacles on starboard side for better training
                    rel_heading_to_collision_pt = np.random.uniform(30, 100)
                else:
                    rel_heading_to_collision_pt = random_sample((30, 130), (230, 330))

            abs_heading = (self.agent.heading + rel_heading_to_collision_pt) % 360
            obs_distance = time_to_collision * obs_velocity  # from target point
            
            # Calculate obstacle starting position
            if obs_type == 3 or obs_type == 4:
                # For overtaking, smaller obs_distance so obstacle is not too close to agent
                scaler = 0.75
            elif obs_type == 4:
                # For crossing, smaller obs_distance spawns obstacles closer
                scaler = 0.5
            else:
                scaler = 1

            obs_xy = collision_xy + obs_distance * scaler * np.array([
                np.sin(np.radians(abs_heading)),
                np.cos(np.radians(abs_heading))
            ])
            
            # Calculate obstacle heading
            obs_heading = (90 - np.degrees(np.arctan2(collision_xy[1] - obs_xy[1], collision_xy[0] - obs_xy[0]))) % 360
            
            if obs_type == 1:  # offset starting position to avoid obstacle
                offset_heading = (obs_heading + np.random.choice([90, -90])) % 360
                obs_xy += obs_safety_radius * 3 * np.array([
                    np.sin(np.radians(offset_heading)),
                    np.cos(np.radians(offset_heading))
                ])

            # Check if position is within the bounds of the ops area
            isInBounds = self.check_in_operational_environment(obs_xy)
            
            # Check that spawned obstacle is not too close to other obstacles
            isTooClose = self.check_if_obstacles_too_close(obs_xy, obs_safety_radius)
            
            # Check that spawned obstacle is not too close to the goal
            isNearGoal = self.check_if_too_close_to_goal(obs_xy)
            
            if isInBounds and not isTooClose and not isNearGoal:
                break

        # Initial heading diff between agent and obstacle for better obstacle classification
        heading_diff = (obs_heading - self.agent.heading + 180) % 360 - 180
        
        # Generate inactive obstacle if impossible to generate within the environment constraints
        if not isInBounds or isTooClose or isNearGoal:
            isActive = False
        else:
            isActive = True
        
        return Obstacle(
            start_pos=obs_xy,
            start_heading=obs_heading,
            start_velocity=obs_velocity,
            safety_radius=obs_safety_radius,
            isActive=isActive,
            initial_heading_diff=heading_diff
        )

    def generate_obstacle(self):

        # Determine obstacle velocity based on motion type
        if (self.obstacle_motion_type == 0) or \
           (self.obstacle_motion_type == 2 and np.random.choice([0, 1]) == 0):
            obstacle = self.generate_static_obstacle()
        else:
            obstacle = self.generate_moving_obstacle()
        
        return obstacle

    def get_agent_state(self):

        # Compute angle diff between agent and goal and distance to goal
        self.agent_dist_to_goal = np.linalg.norm(self.agent.xy - self.goal_pos_xy)
        self.agent_angle_to_goal = self.get_signed_angle_diff(self.agent.xy, self.agent.heading, self.goal_pos_xy)
        angle_diff_rad = np.radians(self.agent_angle_to_goal)

        return np.array([
            self.agent_dist_to_goal / self.max_dist_in_boundary,
            np.sin(angle_diff_rad),
            np.cos(angle_diff_rad),
            self.agent.velocity / self.max_velocity_ms
        ]).astype(np.float32)

    def get_obstacle_state(self, obstacle):
        
        # Compute angle diff between agent and obstacle
        agent_dist_to_obstacle = np.linalg.norm(obstacle.xy - self.agent.xy)
        obstacle_agent_angle_diff = self.get_signed_angle_diff(self.agent.xy, self.agent.heading, obstacle.xy)
        angle_diff_rad = np.radians(obstacle_agent_angle_diff)
        heading_diff = (obstacle.heading - self.agent.heading + 180) % 360 - 180
        heading_diff_rad = np.radians(heading_diff)
        
        # Observation vector for obstacle
        obs_vector = np.array([
            agent_dist_to_obstacle / self.max_dist_in_boundary,
            np.sin(angle_diff_rad),
            np.cos(angle_diff_rad),
            obstacle.velocity / self.max_obs_velocity_ms,
            np.sin(heading_diff_rad),
            np.cos(heading_diff_rad),
            obstacle.sin_initial_heading_diff,
            obstacle.cos_initial_heading_diff
        ]).astype(np.float32)

        # # Get one hot encoding for obstacle type
        # one_hot = np.zeros(6, dtype=np.float32)
        # one_hot[obstacle.type] = 1.0
        
        return obs_vector

    def log_rewards(self, reward, reward_name):
        """Logs each reward/penalty to the rewards_log dict for display in logs table and
        analysis purposes. (Not an essential function)"""
        
        if reward_name == "total_reward": 
            if reward_name not in self.rewards_log: 
                self.rewards_log[reward_name] = 0
            self.rewards_log[reward_name] += reward
        else:
            self.rewards_log[reward_name] = reward 

    def get_head_on_reward(self, obstacle):
        # vec_to_obs = obstacle.xy - self.agent.xy
        # bearing_to_obs = (90 - np.degrees(np.arctan2(vec_to_obs[1], vec_to_obs[0]))) % 360
        # relative_bearing = (bearing_to_obs - self.agent.heading) % 360
        
        vec_to_obs = self.agent.xy - obstacle.xy
        bearing_to_obs = (90 - np.degrees(np.arctan2(vec_to_obs[1], vec_to_obs[0]))) % 360
        relative_bearing_from_obs = (bearing_to_obs - obstacle.heading) % 360

        # if 247.5 <= relative_bearing <= 337.5:
        #     return self.reward_weights_dict["obs_head_on_weightage"]
        # elif 22.5 <= relative_bearing <= 112.5:
        #     return -self.reward_weights_dict["obs_head_on_weightage"]
        # else:
        #     return 0
        
        if 180 <= relative_bearing_from_obs <= 270 and not obstacle.isRewardGiven:
            obstacle.isRewardGiven = True
            return self.reward_weights_dict["obs_head_on_weightage"]
        elif 90 <= relative_bearing_from_obs <= 180 and not obstacle.isRewardGiven:
            obstacle.isRewardGiven = True
            return -self.reward_weights_dict["obs_head_on_weightage"] * 2
        else:
            return 0
    
    def get_crossing_stbd_reward(self, obstacle):
        vec_to_obs = self.agent.xy - obstacle.xy
        bearing_to_obs = (90 - np.degrees(np.arctan2(vec_to_obs[1], vec_to_obs[0]))) % 360
        relative_bearing_from_obs = (bearing_to_obs - obstacle.heading) % 360
        
        heading_diff = (self.agent.heading - self.prev_agent.heading + 180) % 360 - 180
        
        if 0 <= relative_bearing_from_obs <= 90 and not obstacle.isRewardGiven:
            obstacle.isRewardGiven = True
            return -self.reward_weights_dict["obs_crossing_weightage"] * 2
        elif 90 <= relative_bearing_from_obs <= 180 and not obstacle.isRewardGiven:
            obstacle.isRewardGiven = True
            return self.reward_weights_dict["obs_crossing_weightage"]
        else:
            return 0

    def get_overtaking_reward(self, obstacle):
        # vec_to_obs = obstacle.xy - self.agent.xy
        # bearing_to_obs = (90 - np.degrees(np.arctan2(vec_to_obs[1], vec_to_obs[0]))) % 360
        # relative_bearing = (bearing_to_obs - self.agent.heading) % 360

        # if 22.5 <= relative_bearing <= 112.5 and not self.isOvertaken:
        #     self.isOvertaken = True
        #     return self.reward_weights_dict["obs_overtaking_weightage"]
        # elif 247.5 <= relative_bearing <= 337.5 and not self.isOvertaken:
        #     self.isOvertaken = True
        #     return -self.reward_weights_dict["obs_overtaking_weightage"]
        # else:
        #     return 0
        
        vec_to_obs = self.agent.xy - obstacle.xy
        bearing_to_obs = (90 - np.degrees(np.arctan2(vec_to_obs[1], vec_to_obs[0]))) % 360
        relative_bearing_from_obs = (bearing_to_obs - obstacle.heading) % 360
        
        if relative_bearing_from_obs <= 112.5 and not obstacle.isRewardGiven:
            obstacle.isRewardGiven = True
            return -self.reward_weights_dict["obs_overtaking_weightage"] * 2
        elif 247.5 <= relative_bearing_from_obs and not obstacle.isRewardGiven:
            obstacle.isRewardGiven = True
            return self.reward_weights_dict["obs_overtaking_weightage"]
        else:
            return 0

    def get_reward(self, in_ops_env, goal_reached):
        "Calculates the total reward"
        
        self.prev_rewards_log = copy.deepcopy(self.rewards_log)

        # Reward moving towards the goal, penalize moving away from it
        prev_distance = np.linalg.norm(self.prev_agent.xy - self.goal_pos_xy)
        change_in_distance_to_goal = prev_distance - self.agent_dist_to_goal
        distance_change_reward = change_in_distance_to_goal * self.reward_weights_dict["distance_change_weightage"]
        self.log_rewards(distance_change_reward, "distance_change_reward")
        
        # Time penalty
        time_penalty = self.reward_weights_dict["time_penalty_weightage"]
        self.log_rewards(time_penalty, "time_penalty")

        # Penalize exceeding operations environment
        exceed_ops_env_penalty = self.reward_weights_dict["exceed_ops_env_penalty_weightage"] if not in_ops_env else 0
        self.log_rewards(exceed_ops_env_penalty, "exceed_ops_env_penalty") 
        
        # Obstacle-related penalties
        collision_penalty = 0
        too_close_to_obstacle_penalty = 0
        obs_head_on_reward = 0
        obs_crossing_reward = 0
        obs_overtaking_reward = 0
        for obstacle in self.obs_list:
            if obstacle.isActive:
                dist = np.linalg.norm(self.agent.xy - obstacle.xy)

                # Check if agent collided or got too close to obstacle (gradually increase penalty closer to obstacle)
                if dist < 0.2 * obstacle.safety_radius:
                    self.collided = True
                    collision_penalty += self.reward_weights_dict["obs_collision_penalty_weightage"]
                elif dist < 0.3 * obstacle.safety_radius:
                    too_close_to_obstacle_penalty += self.reward_weights_dict["too_close_to_obstacle_penalty_weightage"]
                elif dist < 0.4 * obstacle.safety_radius:
                    too_close_to_obstacle_penalty += 0.7 * self.reward_weights_dict["too_close_to_obstacle_penalty_weightage"]
                elif dist < 0.5 * obstacle.safety_radius:
                    too_close_to_obstacle_penalty += 0.4 * self.reward_weights_dict["too_close_to_obstacle_penalty_weightage"]
                
                # For head on situation, reward agent for crossing on starboard side of obstacle
                if obstacle.type == 2:
                    obs_head_on_reward += self.get_head_on_reward(obstacle)

                # For overtaking situation, reward agent for overtaking on port side of obstacle
                if obstacle.type == 3:
                    obs_overtaking_reward += self.get_overtaking_reward(obstacle)

                # For obstacle crossing on starboard side, agent has to give way
                if obstacle.type == 4:
                    obs_crossing_reward += self.get_crossing_stbd_reward(obstacle)

        self.log_rewards(collision_penalty, "collision_penalty")
        self.log_rewards(too_close_to_obstacle_penalty, "too_close_to_obstacle_penalty")
        self.log_rewards(obs_head_on_reward, "obs_head_on_reward")
        self.log_rewards(obs_crossing_reward, "obs_crossing_reward")
        self.log_rewards(obs_overtaking_reward, "obs_overtaking_reward")

        # Reward for reaching goal
        goal_reward = self.reward_weights_dict["goal_reward_weightage"] if goal_reached else 0
        self.log_rewards(goal_reward, "goal_reward")

        # Final reward
        total_reward = (
            distance_change_reward +
            time_penalty +
            exceed_ops_env_penalty +
            collision_penalty +
            too_close_to_obstacle_penalty +
            obs_head_on_reward +
            obs_crossing_reward +
            obs_overtaking_reward +
            goal_reward
        )
        self.log_rewards(total_reward, "total_reward")

        return total_reward

    def reset(self, seed=None, options=None):

        # Initialise navigation variables (random initialisation)
        # Spawn goal closer to agent for difficulty 1 to ensure interaction with static obstacles
        # Further goal for difficulty 3 to allow sufficient space to overtake
        # if self.difficulty == 1:
        #     self.goal_pos_xy = self.generate_random_coords(50, 100)
        # if self.difficulty == 3:
        #     self.goal_pos_xy = self.generate_random_coords(50, 200)
        # else:
        self.goal_pos_xy = self.generate_random_coords(50, 200)
        self.agent_start_pos_xy = np.array([0.0, 0.0])
        self.agent_start_pos_xy_rel = self.agent_start_pos_xy - self.goal_pos_xy

        # Difficulties 1-4 train agent on collision avoidance, so agent heading is always to the goal
        if 1 <= self.difficulty <= 4:
            diff = self.goal_pos_xy - self.agent_start_pos_xy
            head_on_heading = np.degrees(np.arctan2(diff[0], diff[1]))
            self.initial_heading_degs = head_on_heading
        else:
            self.initial_heading_degs = np.random.uniform(high=360)

        # Adjust environment complexity based on difficulty of environment
        if self.difficulty == 1:
            self.max_spawned_obs = np.random.randint(1, 4)
        elif 2 <= self.difficulty <= 4:
            self.max_spawned_obs = 1
            self.obstacle_motion_type = 1
        elif self.difficulty == 5:
            self.max_spawned_obs = 3
            self.obstacle_motion_type = 1
        
        # Update reward weights dynamically based on difficulty
        # if self.difficulty in [2, 3, 4]:
        #     self.reward_weights_dict["distance_change_weightage"] *= 0.8
        # elif self.difficulty == 4:
        #     self.reward_weights_dict["time_penalty_weightage"] *= 0.8
        
        # Initialise ops environment variables
        self.ops_COG, self.ops_bubble_radius, self.ops_bottom_left, self.ops_top_right, self.max_ops_dist = self.get_operational_environment()
        self.max_dist_in_boundary = 2 * self.ops_bubble_radius
        self.end_time = 4 * self.ops_bubble_radius / self.cruising_speed_ms
        self.elapsed_time = 0
        self.collided = False

        # Initialise agent object
        self.agent = Agent(
            self.agent_start_pos_xy,
            self.initial_heading_degs,
            self.cruising_speed_ms,
            self.max_velocity_ms
        )
        
        # Initialise agent state
        self.state = {'agent': self.get_agent_state()}

        # Initialise obstacle states
        self.state["obstacles"] = []
        self.state["is_active"] = []
        # self.state["obstacle_type"] = []
        self.obs_list = []
        for i in range(self.max_obstacles):

            # Only activate the first "max_spawned_obs" obstacles
            if i < self.max_spawned_obs:
                obstacle = self.generate_obstacle()
            else:
                obstacle = Obstacle(
                    start_pos=np.array([0.0, 0.0]),
                    start_heading=0,
                    start_velocity=0,
                    safety_radius=0,
                    isActive=0,
                    initial_heading_diff=0
                )
            
            self.obs_list.append(obstacle)
            self.state["is_active"].append(obstacle.isActive)
            self.state["obstacles"].append(self.get_obstacle_state(obstacle))
            # self.state["obstacle_type"].append([1, 0, 0, 0, 0])
        
        # Initialise screen-related variables
        self.size_pixels = max(self.metre_to_pixel(self.entity_size), 10)
        self.grid_scale = self.max_dist_in_boundary / self.grid_number
        self.closest_scale = min(
            [1, 2.5, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
            key=lambda x: abs(x - self.grid_scale)
        )
        self.line_length_pixels = int(
            (self.closest_scale / self.max_dist_in_boundary) *
            (self.left_column_width - 2 * self.margins)
        )
        
        return self.state, {}
    
    def step(self, action):
        
        normalized_acc, normalized_yaw_rate = action
        self.acc_ms2 = normalized_acc * self.max_acc_ms2
        self.yaw_rate_degs = normalized_yaw_rate * self.max_yaw_rate_degs

        # Update agent state
        self.prev_agent = copy.deepcopy(self.agent)  # copy previous agent state (mainly for animation)
        self.agent.update(self.acc_ms2, self.yaw_rate_degs, self.time_step)
        self.state['agent'] = self.get_agent_state()

        # Check if agent still in ops env
        in_ops_env = self.check_in_operational_environment(self.agent.xy)

        # Check if agent reached the goal
        if np.linalg.norm(self.agent.xy-self.goal_pos_xy) <= self.proximity_to_goal:
            goal_reached = True
        else:
            goal_reached = False
        
        # Update obstacle state and find number of collisions and instances where agent is too close to obstacle
        self.prev_obs_list = copy.deepcopy(self.obs_list)  # copy previous obs state (mainly for animation)
        for i in range(self.max_spawned_obs):
            obstacle = self.obs_list[i]
            if obstacle.isActive:
                obstacle.update(self.time_step)
                
                # Check if obstacle is outside ops bubble (only relevant for moving obstacles)
                if not self.check_in_operational_environment(obstacle.xy):
                    obstacle = self.generate_obstacle()
                    self.obs_list[i] = obstacle
                    self.state["is_active"][i] = obstacle.isActive

                # Classify non-static obstacles
                if obstacle.velocity > 0:
                    obstacle.type = self.classify_obstacle(obstacle)
                    # # Help agent distinguish between different situations
                    # if obstacle.type == 2:
                    #     self.state["obstacle_type"][i] = [0, 1, 0, 0, 0]
                    # elif obstacle.type == 3:
                    #     self.state["obstacle_type"][i] = [0, 0, 1, 0, 0]
                    # elif obstacle.type == 4:
                    #     self.state["obstacle_type"][i] = [0, 0, 0, 1, 0]
                    # elif obstacle.type == 5:
                    #     self.state["obstacle_type"][i] = [0, 0, 0, 0, 1]
                    # else:
                    #     self.state["obstacle_type"][i] = [1, 0, 0, 0, 0]
                
                self.state["obstacles"][i] = self.get_obstacle_state(obstacle)
        
        # Check if obstacles collided with each other, generate new obstacles to replace collided ones
        collided_obs = self.get_list_of_collided_obstacles()
        for obs in collided_obs:
            obstacle = self.generate_obstacle()
            self.obs_list[obs] = obstacle
            self.state["obstacles"][obs] = self.get_obstacle_state(obstacle)
            self.state["is_active"][obs] = obstacle.isActive

        # Get reward    
        reward = self.get_reward(in_ops_env, goal_reached)

        # Update elapsed time
        self.elapsed_time += self.time_step
        
        # Terminate if goal reached/agent goes out of bounds/collided with obstacle
        terminated = goal_reached or not in_ops_env or self.collided
        truncated = bool(self.elapsed_time >= self.end_time)  # Truncate if elapsed time exceeds end_time
        
        info = {}
        info["episode"] = {"l": self.elapsed_time, "r": self.rewards_log["total_reward"]}

        return self.state, reward, terminated, truncated, info

    # def external_update(self, sensor_data, processed_obs_list, new_goal_pos_longlat):
    #     """Update the environment observation space externally with boat and 
    #     obstacle data. Used in deployment."""

    #     self.goal_pos_xy = np.array(longlat_to_xy(new_goal_pos_longlat))
    #     self.agent.xy = np.array(longlat_to_xy([sensor_data.long, sensor_data.lat]))
    #     self.agent.velocity = np.linalg.norm(sensor_data.velocity)
    #     self.agent.heading = sensor_data.heading
    
    #     self.agent_dist_to_goal = np.linalg.norm(self.agent.xy-new_goal_pos_longlat)
    #     self.agent_angle_to_goal = self.get_signed_angle_diff(self.agent.xy, self.agent.heading, self.goal_pos_xy)

    #     # Update agent state
    #     self.state['agent'] = np.array([
    #                                     self.agent_dist_to_goal / self.max_dist_in_boundary,
    #                                     self.agent_angle_to_goal / 180.0,
    #                                     self.agent.velocity / self.max_velocity_ms,
    #                                     self.agent.heading / 360.0])

    #     # Update Ops env to fit the new goal pos and agent starting position
    #     if not self.check_in_operational_environment(self.agent.xy):
    #         self.update_ops_env(agent_start_pos_longlat= [sensor_data.long, sensor_data.lat], 
    #                             new_goal_pos_longlat= new_goal_pos_longlat)

    #     # Process detected obstacles
    #     for obs in processed_obs_list:
            
    #         obs_xy = obs.xy_abs
    #         if np.nan in obs_xy:
    #             continue
    #         obs_velocity = np.linalg.norm(obs.velocity_abs)
    #         obs_heading = np.rad2deg(np.arctan2(obs.velocity_abs[1], obs.velocity_abs[0]))
    #         tracker_id = obs.id
            
    #         if tracker_id not in self.obs_to_tracker_id_dict.values(): 
    #             # Check if any ids are available
    #             have_id = False
    #             for i in range(1, self.max_spawned_obs+1):
    #                 if self.obs_to_tracker_id_dict[i] == -1: 
    #                     self.obs_to_tracker_id_dict[i] = tracker_id
    #                     have_id = True
    #                     break
            
    #             if not have_id: continue # Skip the object if no ids availabele
            
    #         # Get the corresponding obstacle id from the tracker id
    #         for i in self.obs_to_tracker_id_dict.keys():
    #             if self.obs_to_tracker_id_dict[i] == tracker_id:  
    #                 obs_id = i
            
    #         # Determine obs safety radius
    #         if obs.size <= 10: obs_size = "small"
    #         elif obs.size <= 20: obs_size = "medium"
    #         else: obs_size = "large"
            
    #         # Check if obstacle exceeded ops env
    #         if np.linalg.norm(obs_xy - self.ops_COG) > self.ops_bubble_radius:
    #             self.state[f'obs{obs_id}_active'] = 0 # Deactivate obstacle

    #         self.state[f'obs{obs_id}_active'] = 1
    #         self.state[f'obs{obs_id}'] = np.array([
    #                                         np.linalg.norm(obs_xy-self.agent.xy) / self.max_dist_in_boundary,
    #                                         self.get_signed_angle_diff(self.agent.xy, self.agent.heading, self.goal_pos_xy) / 180.0,
    #                                         obs_velocity / self.max_obs_velocity_ms,
    #                                         obs_heading / 360.0,
    #                                         self.safety_radius_dict[obs_size], # Need to normalise
    #                                         ])
    #         self.state[f'obs{obs_id}_type'] = self.classify_obstacle(Obstacle(obs_xy, 
    #                                                                             obs_heading, 
    #                                                                             obs_velocity, 
    #                                                                             self.safety_radius_dict[obs_size],
    #                                                                             self.state[f'obs{obs_id}_active']))
            
    #     for i in range(1, self.max_spawned_obs+1):
    #         # Deactivate the obstacle if it is currently not detected
    #         if self.obs_to_tracker_id_dict[i] not in [obs.id for obs in processed_obs_list]:      
    #             self.state[f'obs{i}_active'] = 0 
            
    #         # Track how long the obstacle ID has been inactive
    #         if self.state[f'obs{i}_active'] == 0: self.obs_dead_time_list[i-1] += 1

    #         # Reset the tracker
    #         if self.state[f'obs{i}_active'] == 1: self.obs_dead_time_list[i-1] = 0
                
    #         # Free the obstacle ID if it has not been active
    #         if self.obs_dead_time_list[i-1] > 10: self.obs_to_tracker_id_dict[i-1] = -1
        
    #     return self.state, self.agent_dist_to_goal
        
    # def update_ops_env(self, agent_start_pos_longlat, new_goal_pos_longlat):
    #     """Update the ops env with the new agent start pos and goal pos. Used in deployment"""
        
    #     self.goal_pos_xy = np.array(longlat_to_xy(new_goal_pos_longlat))
    #     self.agent_start_pos_xy = np.array(longlat_to_xy(agent_start_pos_longlat))
    #     self.agent_start_pos_xy_rel = self.agent_start_pos_xy - self.goal_pos_xy
        
    #     self.ops_COG, self.ops_bubble_radius, self.ops_bottom_left, self.ops_top_right, self.max_ops_dist = self.get_operational_environment()

    #     self.grid_scale = self.ops_bubble_radius * 2 / self.grid_number  # metres
        
    #     self.closest_scale = min(
    #         [1, 2.5, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
    #         key=lambda x: abs(x - self.grid_scale),
    #     )
        
    #     # Set end time for truncation
    #     self.end_time = (
    #         4 * self.ops_bubble_radius / self.cruising_speed_ms 
    #     ) # seconds

    # def get_next_state_and_update_agent(self, acc_ms2, yaw_rate_degs):
    #     "Returns self.state['agent'] after taking action"
        
    #     ### DAVID's EDIT START
        
    #     # Get next position
    #     self.agent.xy += (self.agent.velocity * self.time_step * 
    #         np.array(
    #             [np.cos(np.deg2rad(compass_to_math_angle(self.agent.heading))), 
    #              np.sin(np.deg2rad(compass_to_math_angle(self.agent.heading)))]
    #             )
    #         )

    #     # Get next velocity
    #     self.agent.velocity += acc_ms2 * self.time_step
    #     self.agent.velocity = np.clip(self.agent.velocity, 0, self.max_velocity_ms) 
        
    #     # Simulate drag to agent's linear motion *(not accurate representation of actual drag)
    #     if acc_ms2 == 0 and self.agent.velocity > 0:
    #         drag_coefficient = 0.1
    #         wetted_area = 2 * 0.8 # metre ^2
    #         water_density = 1000 # kg/m^3
    #         mass = 40 # kg
            
    #         drag_force = 0.5 * water_density * drag_coefficient * wetted_area * self.agent.velocity**2
    #         deceleration = drag_force / mass
            
    #         self.agent.velocity = max(self.agent.velocity - deceleration * self.time_step, 0)
    #         if self.agent.velocity <= 0.1: self.agent.velocity = 0
            
    #     self.agent.heading += self.time_step * yaw_rate_degs
    #     self.agent.heading = self.agent.heading % 360  # Ensure heading is within the range [0, 360]
        
    #     # Normalise agent state
    #     distance_to_goal_norm = self.agent.get_dist_to_goal(self.goal_pos_xy) / self.max_ops_dist_scalar
    #     angle_to_goal_norm = self.agent.get_angle_diff_to_goal(self.goal_pos_xy) / 180.0
    #     velocity_norm = self.agent.velocity/self.max_velocity_ms
    #     heading_norm = self.agent.heading/360.0

    #     return np.array(
    #         [distance_to_goal_norm, 
    #          angle_to_goal_norm, 
    #          velocity_norm, 
    #          heading_norm,
    #          ], dtype=np.float32
    #     )
    #     ### DAVID's EDIT END

    # def get_angle_to_goal(self, agent_xy, goal_pos_xy, agent_heading):
    #     """Get the angle difference between the agent's heading and the goal position"""
        
    #     # Get heading of goal relative to North from agent 
    #     goal_heading = heading_to_goal(xy_to_longlat(agent_xy), 
    #                                    xy_to_longlat(goal_pos_xy)) 
    #     angle_diff = (goal_heading - agent_heading) % 360 # Restrict angles to [0, 360]
    #     angle_diff = min(angle_diff, 360 - angle_diff) # Calculate the smallest angle difference between agent and goal heading

    #     return angle_diff
    
    # @staticmethod
    # def power_reward_func(pt1, pt2, cal_pt, concavity, power):
    #     if not (min(pt1[0], pt2[0]) <= cal_pt <= max(pt1[0], pt2[0])):
    #         raise ValueError("cal_pt must be between x1 and x2")
        
    #     if pt2[1] > pt1[1]: 
    #         big_pt = pt2
    #         small_pt = pt1
    #     else: 
    #         big_pt = pt1
    #         small_pt = pt2

    #     if concavity == 'down':
    #         ref_pt = big_pt
    #         final_pt = small_pt
    #     elif concavity == 'up':
    #         ref_pt = small_pt
    #         final_pt = big_pt
    #     else:
    #         raise TypeError('concavity is either up or down')

    #     x_2 = final_pt[0] - ref_pt[0]
    #     x = cal_pt - ref_pt[0]
    #     a_2 = final_pt[1] - ref_pt[1]

    #     if type(power) is int and power >= 1:
    #         reward = a_2 * (x/x_2)**power + ref_pt[1]
    #     else:
    #         raise TypeError('power must be an integer greater than 1')
    #     return reward

    def xy_to_pixel(self, xy):
        "Converts xy to pixel coordinates"

        # Scale the coordinates to fit within the screen dimensions
        pixel_x = int(
            self.margins
            + ((xy[0] - self.ops_bottom_left[0]) / (2 * self.ops_bubble_radius))
            * (self.left_column_width - 2 * self.margins)
        )
        pixel_y = int(
            self.margins
            + ((self.ops_top_right[1] - xy[1]) / (2 * self.ops_bubble_radius))
            * (self.left_column_width - 2 * self.margins)
        )

        return np.array([pixel_x, pixel_y])

    def metre_to_pixel(self, metres):
        "Converts metres to number of pixels"
        return int(
            metres
            / (2 * self.ops_bubble_radius)
            * (self.left_column_width - 2 * self.margins)
        )

    def render(self):

        pygame.display.set_caption("Maritime Environment")
        self.clock = pygame.time.Clock()
        
        # Predefined fonts for pygame screen
        self.large_font = pygame.font.SysFont("segoeui", 18, bold=True)
        self.medium_font = pygame.font.SysFont("segoeui", 14, bold=True)
        self.small_font = pygame.font.Font(None, 20)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return

        ### DAVID's EDIT START ###
        if self.simulation: # interpolate positions of entites during simulated lag

            for j in range(self.steps):
                
                # Draw the dark blue background in the drawing area
                pygame.draw.rect(self.screen, DARK_BLUE, self.drawing_area)        
                self.draw_grid()
                self.draw_goal()
                
                # Drawing (dynamic agent)
                self.draw_agent(interpolate=True, j=j)
                
                # Drawing (dynamic obstacles)
                self.draw_obstacles(interpolate=True, j=j)
                
                # Draw static elements
                self.draw_margins()
                self.draw_game_status()
                self.draw_north_arrow()
                self.draw_display_scale()
                self.draw_obs_types()
                pygame.draw.circle(
                    self.screen, 
                    LIGHT_GREY, 
                    self.xy_to_pixel(self.ops_COG), 
                    self.metre_to_pixel(self.ops_bubble_radius), 
                    width=self.linewidth_pixels
                ) # Draw ops bubble radius circle
                
                self.draw_agent_properties()
                self.draw_reward_logs_table()
                # self.draw_wasd()
                
                # Update screen
                pygame.display.flip() 
                
                # Capture the screen
                frame = pygame.surfarray.array3d(self.screen)
                frame = np.transpose(frame, (1, 0, 2))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if self.record: self.vid_holder.write(frame)
                
                self.clock.tick(self.display_rate)
                
                # Return pygame screen as an image 
                return frame

        else: # show live positions of entities
            
           # Draw the dark blue background in the drawing area
            pygame.draw.rect(self.screen, DARK_BLUE, self.drawing_area)        
            self.draw_grid()
            self.draw_goal()
            
            # *** Agent and obstacles are plotted on their exact pos, no interpolation 
            # Drawing (dynamic agent)
            self.draw_agent()
            # Drawing (dynamic obstacles)
            self.draw_obstacles()
            
            # Draw static elements
            self.draw_margins()
            self.draw_game_status()
            self.draw_north_arrow()
            self.draw_display_scale()
            self.draw_obs_types()
            pygame.draw.circle(
                self.screen, 
                LIGHT_GREY, 
                self.xy_to_pixel(self.ops_COG), 
                self.metre_to_pixel(self.ops_bubble_radius), 
                width=self.linewidth_pixels
            ) # Draw ops bubble radius circle
            
            self.draw_agent_properties()
            # self.draw_reward_logs_table()

            # Update screen
            pygame.display.flip() 
            
            # Capture the screen
            frame = pygame.surfarray.array3d(pygame.display.get_surface())
            frame = np.transpose(frame, (1, 0, 2))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if self.record: self.vid_holder.write(frame)
            
            self.clock.tick(self.display_rate)
            return frame

    def draw_agent(self, interpolate=False, j=None):
        "Draws agent"
        
        ### DAVID's EDIT START
        
        if interpolate:
            interpolated_agent_pos_arr = self.interpolate_pixel_pos(
                self.xy_to_pixel(self.prev_agent.xy),
                self.xy_to_pixel(self.agent.xy),
            )
            pixel_xy = interpolated_agent_pos_arr[j]
        else:
            pixel_xy = self.xy_to_pixel(self.agent.xy)      
            
        # Track the agent's position for drawing the beeline
        self.agent_pixel_pos_deque.append(pixel_xy)
        if len(self.agent_pixel_pos_deque) > 100: self.agent_pixel_pos_deque.popleft() 
        
        rect = pygame.Rect(
            pixel_xy[0] - self.size_pixels // 2,
            pixel_xy[1] - self.size_pixels // 2,
            self.size_pixels,
            self.size_pixels,
        )
        # Draw the square
        pygame.draw.rect(self.screen, YELLOW, rect, width=self.linewidth_pixels)

        arrow_length = max(
            min(
                self.metre_to_pixel(self.agent.velocity) * VELOCITY_ARROW_SCALE, 
                MAX_ARROW_LENGTH_PIXELS
            ),  
            MIN_ARROW_LENGTH_PIXELS)
        
        # Calculate the end point of the arrow
        end_xy = (pixel_xy + arrow_length * np.array([
            np.sin(np.radians(self.agent.heading)),
            -np.cos(np.radians(self.agent.heading))
        ])).astype(int)

        # Draw arrow
        pygame.draw.line(
            self.screen,
            YELLOW,
            pixel_xy,
            end_xy,
            width=self.linewidth_pixels,
        )

        # Detection radius circle
        pygame.draw.circle(
            self.screen,
            YELLOW, 
            pixel_xy,
            self.metre_to_pixel(self.detection_radius),
            width=self.linewidth_pixels,
        )

        # Draw a beeline behind the agent to visualise its motion
        for i, pos in enumerate(list(self.agent_pixel_pos_deque)[::-1]):
            if i % 5 == 0:
                pygame.draw.circle(
                    self.screen, LIGHT_GREY, pos, self.linewidth_pixels,
                )

    def draw_obstacles(self, interpolate=False, j=None):
        """Draws all obstacles. 
        Set interpolate=True when simulating lag. 
        j: display_rate step"""
        
        # Only draw active obstacles
        for i in range(self.max_spawned_obs):
            obs = self.obs_list[i]
            if obs.isActive:
                if interpolate:  # Interpolate obs position
                    if self.prev_obs_list[i].isActive == 1:  # Interpolate if obs has previous position
                        interpolated_obs_pos_arr = self.interpolate_pixel_pos(
                            self.xy_to_pixel(self.prev_obs_list[i].xy),
                            self.xy_to_pixel(obs.xy)
                        )
                        pixel_xy = interpolated_obs_pos_arr[j]
                    else:
                        continue
                else:
                    pixel_xy = self.xy_to_pixel(obs.xy)
                    
                # Check if the xy position is within the map_area (pygame.rect)
                if self.drawing_area.collidepoint(pixel_xy):
                    colour = self.colours_dict[obs.type][1]
                    obs_square = pygame.Rect(
                        pixel_xy[0] - self.size_pixels // 2,
                        pixel_xy[1] - self.size_pixels // 2,
                        self.size_pixels,
                        self.size_pixels
                    )
                    # Draw the square
                    pygame.draw.rect(self.screen, colour, obs_square, width=self.linewidth_pixels)

                    # Arrow length is proportional to velocity of object (in pixels)
                    arrow_length = max(
                        min(
                            self.metre_to_pixel(obs.velocity) * VELOCITY_ARROW_SCALE, 
                            MAX_ARROW_LENGTH_PIXELS
                        ),
                        MIN_ARROW_LENGTH_PIXELS
                    ) 

                    # Calculate the end point of the arrow
                    end_xy = (pixel_xy + arrow_length * np.array([
                        np.sin(np.radians(obs.heading)),
                        -np.cos(np.radians(obs.heading))
                    ])).astype(int)

                    # Draw arrow if obstacle is not stationary
                    if obs.velocity != 0:
                        pygame.draw.line(
                            self.screen,
                            colour,
                            pixel_xy,
                            end_xy,
                            width=self.linewidth_pixels,
                        )

                    SR_pixels = self.metre_to_pixel(obs.safety_radius)

                    # Draw the safety radius circle 
                    pygame.draw.circle(
                        self.screen,
                        colour,
                        pixel_xy,
                        SR_pixels,
                        width=self.linewidth_pixels,
                    )

                    ### DAVID's EDIT START
                    # Draw the obstacle id 
                    font = pygame.font.SysFont(None, 20) 
                    obj_id_text = font.render(str(i), 
                                            True, 
                                            colour)
                    text_w, text_h = obj_id_text.get_size()
                    self.screen.blit(obj_id_text, (pixel_xy[0]+self.size_pixels//2+5, pixel_xy[1]-text_h//2-1))
                    ### DAVID's EDIT END

    def draw_goal(self):
        "Draws goal"
        pixel_xy = self.xy_to_pixel(self.goal_pos_xy)
        pygame.draw.circle(
            self.screen, GREEN, pixel_xy, self.size_pixels // 2
        )  # Green goal
    
        ### DAVID's ADDITION START
        pygame.draw.circle(
            self.screen, GREEN, pixel_xy, self.metre_to_pixel(self.proximity_to_goal), width=self.linewidth_pixels
        )  # Proximity circle
        ### DAVID's ADDITION END ###
        
    def draw_grid(self):
        "Draw gridlines"
        # Draw vertical grid lines
        for x in range(
            self.drawing_area.left,
            self.drawing_area.right + self.grid_size,
            self.grid_size,
        ):
            pygame.draw.line(
                self.screen,
                LIGHT_GREY,
                (x, self.drawing_area.top),
                (x, self.drawing_area.bottom),
            )
        # Draw horizontal grid lines
        for y in range(
            self.drawing_area.top,
            self.drawing_area.bottom + self.grid_size,
            self.grid_size,
        ):
            pygame.draw.line(
                self.screen,
                LIGHT_GREY,
                (self.drawing_area.left, y),
                (self.drawing_area.right, y),
            )

    def draw_display_scale(self):
        "Draws the display scale in metres at bottom right corner"

        x = self.left_column_width - self.margins
        y = self.left_column_width - (4 * self.margins // 5)

        # Draw the scale
        text_surface = self.medium_font.render(f"{self.closest_scale}m", True, WHITE)
        w, h = text_surface.get_size()
        x -= w
        self.screen.blit(text_surface, (x, y))  

        # Draw the horizontal line above the scale text
        line_end_x = x - 10  # Position the line next to the text
        line_start_x = line_end_x - self.line_length_pixels
        line_y = y+h/2
        pygame.draw.line(
            self.screen,
            WHITE,
            (line_start_x, line_y),
            (line_end_x, line_y),
            width=self.left_column_width // 200,
        )

    ### DAVID's ADDITION START
    def draw_game_status(self):
        """Displays the collision status of the agent with obstacles"""
        
        x, y = self.margins, self.left_column_width // 100
        word_spacing = 10
        
        collision_status = "Safe"
        collided_obs_text = ""
        color = GREEN
        for i in range(1, self.max_obstacles+1):

            if self.collision_flags[i-1] == True: #  check if collided with any obstacles
                collision_status = "Collided" 
                collided_obs_text + f"{i} "
                color = RED
                
        w, h = self.draw_text("Collision Status: ", (x, y),self.large_font) # Position the text at the top-left corner
        x += w
        w, h = self.draw_text(collision_status, (x, y),self.large_font, color)
        x += w+word_spacing
        if collision_status == "Collided":
            w, h = self.draw_text(f"Obstacles: {collided_obs_text}", (x, y),self.large_font)
            x += w
        self.draw_text(f"Time Elapsed: {self.elapsed_time:.1f}/{self.end_time:.1f}s Distance to Goal: {self.agent_dist_to_goal:.0f}m Angle to Goal: {self.agent_angle_to_goal:.0f}°", 
                (x, y),self.large_font)
    ### DAVID's ADDITION END

    def draw_north_arrow(self):
        "Draws north arrow"
        arrow_size = self.left_column_width // 50
        arrow_pos = (self.left_column_width - 1.5 * arrow_size, self.margins + arrow_size)
        arrow_vertices = [
            (arrow_pos[0], arrow_pos[1] - arrow_size),  # Top
            (
                arrow_pos[0] - arrow_size // 2,
                arrow_pos[1] + arrow_size // 2,
            ),  # Bottom left
            (arrow_pos[0], arrow_pos[1] - arrow_size // 6),  # bottom kink
            (
                arrow_pos[0] + arrow_size // 2,
                arrow_pos[1] + arrow_size // 2,
            ),  # Bottom right
        ]
        pygame.draw.polygon(self.screen, WHITE, arrow_vertices)

        # Label the arrow with "N"
        self.draw_text("N",             
            (
                arrow_pos[0] - 7,
                arrow_pos[1] + 14,
            ), # Position below the arrow
            self.large_font) 

    def draw_margins(self):
        "Draws black margins (Covers parts of the obs/agent that is outside the map)"
        ### DAVID's EDIT START ###
        pygame.draw.rect(
            self.screen, BLACK, (0, 0, self.screen_width, self.margins)
        )  # Top margin
        pygame.draw.rect(
            self.screen, BLACK, (0, 0, self.margins, self.screen_height)
        )  # Left margin
        pygame.draw.rect(
            self.screen,
            BLACK,
            (0, self.screen_height - self.margins + 1, self.screen_width, self.margins),
        )  # Bottom margin
        pygame.draw.rect(
            self.screen,
            BLACK,
            (self.left_column_width - self.margins, 0, self.screen_width-(self.left_column_width-self.margins)
             , self.screen_height),
        )  # Right margin
        
        ### DAVID's EDIT END ###

    def draw_obs_types(self):
        "Displays colour-coded text for obstacle types"

        x = self.margins  # Initial x position for text
        y = self.left_column_width - (self.margins//3 * 2)  # y position for text

        for _, value in self.colours_dict.items():
            # Position the text at the bottom left corner
            w, h = self.draw_text(value[0], (x,y),self.medium_font, value[1])

            # Update x position for the next piece of text
            x += w + 10  # Add some space between texts

    def draw_agent_properties(self):
        "Display agent's state information"

        ### DAVID's EDIT START ###
        agent_velocity_knots = mstoknots(self.agent.velocity)
        agent_heading = self.agent.heading
        agent_longlat = xy_to_longlat(self.agent.xy)

        x = self.left_column_width
        y = self.screen_height//8
        row_spacing = 10
        
        w, h = self.draw_text(f"Longitude: {agent_longlat[0]:.6f}°", 
                       (x, y),self.large_font)
        y += row_spacing+h
        w, h = self.draw_text(f"Latitude: {agent_longlat[1]:.6f}°", 
                       (x, y),self.large_font)   
        y += row_spacing+h
        w, h = self.draw_text(f"Velocity: {agent_velocity_knots:.1f} knots", 
                       (x, y),self.large_font)   
        y += row_spacing+h
        w, h = self.draw_text(f"Heading: {agent_heading:.0f}°", 
                       (x, y),self.large_font)  
        y += row_spacing+h
        w, h = self.draw_text(f"Acceleration: {self.acc_ms2:.4f}m/s^2", 
                       (x, y),self.large_font)
        y += row_spacing+h 
        w, h = self.draw_text(f"Yaw Rate: {self.yaw_rate_degs:.2f}°/s", 
                       (x, y),self.large_font)  
        ### DAVID's EDIT END ###
    
    ### DAVID's ADDITION START ###
    # Draw an arrow to represent increase or decrease in reward value
    def draw_change_arrow(self, x, y, direction, color): 
        y+=1
        if direction == "up":
            points = [(x, y), (x + 5, y+10), (x - 5, y+10)]
        elif direction == "down":
            points = [(x, y+10), (x + 5, y), (x - 5, y)]
        pygame.draw.polygon(self.screen, color, points)

    # Render the rewards table
    def draw_reward_logs_table(self):
        
        start_pos=(self.left_column_width, 3*self.screen_height//8)
        x, y = start_pos
        row_spacing=20
        value_x = x + 200
        
        # Draw the table header
        self.draw_text("Rewards", (x,y), self.large_font)
        pygame.draw.line(self.screen, WHITE, (x, y+30), (self.left_column_width+300-50, y+30), 2)

        y += 40

        # Render each reward
        for reward_name, value in self.rewards_log.items():
            if reward_name == "total_reward":  # Skip to render total reward at the end
                continue

            # Draw reward name
            self.draw_text(reward_name, (x,y), self.small_font, LIGHT_GREY)
            w, h = self.draw_text(round(value, 2), (value_x,y), self.small_font, WHITE)
            arrow_x = value_x+w+10

            # Check if value has changed and draw an arrow
            if reward_name in self.prev_rewards_log:
                if value > self.prev_rewards_log[reward_name]:
                    self.draw_change_arrow(arrow_x, y, "up", GREEN)  # Green for increase
                elif value < self.prev_rewards_log[reward_name]:
                    self.draw_change_arrow(arrow_x, y, "down", RED)  # Red for decrease

            y += row_spacing

        if "total_reward" in self.rewards_log:
            # Render total self.rewards_log at the bottom
            total_reward_value = self.rewards_log["total_reward"]

            # Draw total rewards
            self.draw_text("Total Reward", (x,y), self.small_font, YELLOW)
            w, h = self.draw_text(round(total_reward_value, 2), (value_x, y), self.small_font, YELLOW)
            arrow_x = value_x + w + 10

            # Check if total rewards changed and draw an arrow
            if "total_reward" in self.prev_rewards_log:
                if total_reward_value > self.prev_rewards_log["total_reward"]:
                    self.draw_change_arrow(arrow_x, y, "up", GREEN)  # Green for increase
                elif total_reward_value < self.prev_rewards_log["total_reward"]:
                    self.draw_change_arrow(arrow_x, y, "down", RED)  # Red for decrease

    def draw_text(self, text, xy, font, color=WHITE, background=None):
        """Draws text on the screen. Returns the width and height of the text. (w,h)"""
        text_surface = font.render(str(text), True, color, background)
        self.screen.blit(text_surface, xy)  
        return text_surface.get_size()
    
    def draw_wasd(self):
        """Draw WASD keys onto map"""

        key_size = 40
        space = 10
        # Top left position of W key
        x1, y1 = (self.left_column_width-self.margins-2*space-2*key_size, self.screen_height-self.margins-2*space-2*key_size)
        
        # Define key positions
        keys = {
            "W": (x1, y1),
            "A": (x1-space-key_size, y1+space+key_size),
            " ": (x1, y1+space+key_size),
            "D": (x1+space+key_size, y1+space+key_size),
        }
                
        for key, (x, y) in keys.items():
            pygame.draw.rect(self.screen, BLACK, (x, y, key_size, key_size), border_radius=10)
            self.draw_text(key, (x+15,y+10), self.medium_font, WHITE)
    ### DAVID's ADDITION END ###
       
    def interpolate_pixel_pos(self, pixels_start, pixels_end):
        "Interpolates between (pixel_x_start, pixel_y_start) and pixels_end = (pixel_x_end, pixel_y_end), returns array of tuples"
        x = np.linspace(pixels_start[0], pixels_end[0], self.steps)
        y = np.linspace(pixels_start[1], pixels_end[1], self.steps)
        return [(int(x[i]), int(y[i])) for i in range(self.steps)]

    def close(self):
        # self.out.release()
        pygame.quit()
        cv2.destroyAllWindows()
