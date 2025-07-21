import time

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from params import *
from model_funcs import *
from curriculum import CurriculumCallback

NUM_ENVS = 4
TRAINING_TIMESTEPS = 1_000_000

MODEL_PATH = "RL_training/sac_model"

# Wrap the environment in a vectorized environment
vec_env = make_vec_env(create_env, n_envs=NUM_ENVS)

model = SAC("MultiInputPolicy",
            env=vec_env,
            device="cpu")

start_time = time.time()  # Start the timer

model.learn(
    total_timesteps=TRAINING_TIMESTEPS,
    progress_bar=True,
    callback=CurriculumCallback(TRAINING_TIMESTEPS)
)
model.save(MODEL_PATH)

end_time = time.time() # End the timer

total_time = end_time - start_time
print(f"\nTotal time taken for training: {total_time//3600}hrs, {total_time%3600//60}mins & {total_time%3600%60}secs")