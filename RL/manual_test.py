from stable_baselines3 import SAC
from params import *
from model_funcs import *

MODEL_PATH = "RL_training/sac_model_multi.zip"

# Create the environment
env = create_env(difficulty=6, record=True)

# Load the trained model
model = SAC.load(MODEL_PATH)

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