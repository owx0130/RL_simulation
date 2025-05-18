"""
visualize_env.py

This module is responsible for visualizing the maritime simulation.
It sets up the environment, runs the simulation loop, and renders the environment.

The script initializes the custom environment `MyEnv`. It then runs the simulation loop, rendering the environment at each step.

Usage:
    Run this script to visualize the maritime simulation environment. Press Esc key to exit the simulation. If uninterrupted simulation will
    run until self.end_time is reached or agent has reached the goal. 
    Use the W, A & D keys to control the agent to test the simulation environment.

    Run this script in src/main/

Example:
    $ python visualize_env.py
"""

import pygame

from params import *
from model_funcs import *
    
if __name__ == "__main__":

    env = create_env(
        rec=False,                     
        vid_name="Environment_demo",
        sim_status=True
    )

    # Create the environment
    env.reset()
    terminated = False
    truncated = False
    
    total_reward = 0
    
    while not (terminated or truncated):
        
        # Control the agent's position using W, A & D keys
        for event in pygame.event.get():
            
            # Check keys continuously
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w] or keys[pygame.K_s]:  
                if keys[pygame.K_w]: # Increase acceleration
                    acceleration_normalized = 1
            else:
                acceleration_normalized = 0
            if keys[pygame.K_a] or keys[pygame.K_d]: 
                if keys[pygame.K_a]:  # Increase yaw to the left
                    yaw_rate_normalized = -1
                if keys[pygame.K_d]:  # Increase yaw to the right
                    yaw_rate_normalized = 1
            else:
                yaw_rate_normalized = 0

        acceleration_normalized = max(0, min(acceleration_normalized, 1))
        yaw_rate_normalized = max(-1, min(yaw_rate_normalized, 1))
        
        action = (acceleration_normalized, yaw_rate_normalized)
        # action = env.action_space.sample()  # Sample random action

        state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        
        frame = env.render()  # Render the environment

    print("Total Rewards: ", total_reward)

    env.close()

