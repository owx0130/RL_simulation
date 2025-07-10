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

