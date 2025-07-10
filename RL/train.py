import time
import shutil

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from params import *
from model_funcs import *
from curriculum import CurriculumCallback

TRAINING_TIMESTEPS = 1_000_000

MODEL_NAME = "simple_nav"

# Directories
MODEL_DIR = os.path.join(os.getcwd(), "RL_training", MODEL_NAME) # /training/<model_name>/
LOG_DIR = os.path.join(MODEL_DIR, "logs") # /training/<model_name>/logs/

if os.path.exists(MODEL_DIR):
    shutil.rmtree(MODEL_DIR)

# Wrap the environment in a vectorized environment
NUM_ENVS = 4
vec_env = make_vec_env(create_env, n_envs=NUM_ENVS)

base_env = create_env()
model = SAC("MultiInputPolicy",
            env=vec_env,
            device="cpu")

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
