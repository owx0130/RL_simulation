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

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env 
from stable_baselines3.common.vec_env import DummyVecEnv

from hybrid_callback import HybridCallback
from params import *
from model_funcs import *

TRAINING_TIMESTEPS = 150_000

MODEL_NAME = "simple_nav"

# Directories
MODEL_DIR = os.path.join(os.getcwd(), "RL_training", MODEL_NAME) # /training/<model_name>/
LOG_DIR = os.path.join(MODEL_DIR, "logs") # /training/<model_name>/logs/

# Wrap the environment in a vectorized environment
NUM_ENVS = 4
vec_env = make_vec_env(create_env, n_envs=NUM_ENVS, vec_env_cls=DummyVecEnv) # Create the vectorized environment

model = PPO("MultiInputPolicy", env=vec_env, verbose=1, tensorboard_log=LOG_DIR)

print(f"\nRun command to view Tensorboard logs: tensorboard --logdir={LOG_DIR}\n")

start_time = time.time() # Start the timer

model.learn(
    total_timesteps=TRAINING_TIMESTEPS,
    progress_bar=True,
    callback=HybridCallback(
        n_eval_episodes=3,
        eval_freq=16384,
        save_freq=16384,
        save_path=MODEL_DIR,
        log_path=LOG_DIR,
        render=False
    )
)

end_time = time.time() # End the timer

total_time = end_time - start_time
print(f"\nTotal time taken for training: {total_time//3600}hrs, {total_time%3600//60}mins & {total_time%3600%60}secs")

print(f"\nRun command to view Tensorboard logs: tensorboard --logdir={LOG_DIR}\n")
