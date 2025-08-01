from stable_baselines3 import SAC
from params import *
from model_funcs import *

MODEL_PATH = "RL_training/sac_model_multi_2.zip"

# Create the environment and indicate desired difficulty for testing
env = create_env(difficulty=6, record=True)

# Load the trained model
model = SAC.load(MODEL_PATH)

# Reset the environment and get the initial observation
obs, _ = env.reset()

# Continuously step throuh the environment to test the model
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