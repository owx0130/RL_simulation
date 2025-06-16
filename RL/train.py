"""
train.py

This script trains a Proximal Policy Optimization (PPO) agent using a custom environment defined in `env.py`.

The PPO agent is defined as a neural network with three fully connected layers. The training loop runs for a specified number of epochs, collecting data from the environment, computing losses, and updating the agent's parameters. The trained model is saved at the end of the training process.

Classes:
    PPOAgent: Defines the neural network architecture for the PPO agent.

Functions:
    forward(state): Performs a forward pass through the network.

Usage:
    Run this script to train the PPO agent. 

Example:
    $ python train.py

Tensorboard:
   $ tensorboard --logdir=<logdir>
"""

import time
import shutil

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env

from params import *
from model_funcs import *
from curriculum import CurriculumCallback

TRAINING_TIMESTEPS = 720_000

MODEL_NAME = "simple_nav"

# Directories
MODEL_DIR = os.path.join(os.getcwd(), "RL_training", MODEL_NAME) # /training/<model_name>/
LOG_DIR = os.path.join(MODEL_DIR, "logs") # /training/<model_name>/logs/

if os.path.exists(MODEL_DIR):
    shutil.rmtree(MODEL_DIR)

# Wrap the environment in a vectorized environment
NUM_ENVS = 6
vec_env = make_vec_env(create_env, n_envs=NUM_ENVS)
model = SAC("MultiInputPolicy", env=vec_env)

print(f"\nRun command to view Tensorboard logs: tensorboard --logdir={LOG_DIR}\n")

start_time = time.time()  # Start the timer

model.learn(
    total_timesteps=TRAINING_TIMESTEPS,
    progress_bar=True,
    callback=CurriculumCallback(TRAINING_TIMESTEPS)
)
model.save("RL_training/sac_model")

end_time = time.time() # End the timer

total_time = end_time - start_time
print(f"\nTotal time taken for training: {total_time//3600}hrs, {total_time%3600//60}mins & {total_time%3600%60}secs")

print(f"\nRun command to view Tensorboard logs: tensorboard --logdir={LOG_DIR}\n")
