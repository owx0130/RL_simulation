import optuna
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from params import *
from model_funcs import *

def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512, 1024])
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    tau = trial.suggest_float('tau', 0.005, 0.05)

    # Create vectorized env (use your custom env here)
    env = make_vec_env(create_env, n_envs=2)

    # Build model
    model = SAC(
        "MultiInputPolicy",
        env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
    )

    # Train briefly
    model.learn(total_timesteps=100_000)

    # Evaluate performance
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
    env.close()

    return mean_reward

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print("Best trial:")
print(study.best_trial)