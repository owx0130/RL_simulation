"""
visualize_model.py

This script loads a trained Proximal Policy Optimization (PPO) agent and uses it to interact with and render a custom environment defined in `env.py`.

The script creates an instance of the PPO agent, loads the saved model parameters, and runs the agent in the environment, rendering each step.

Usage:
    Run this script to visualize the performance of the trained PPO agent.

Example:
    $ python visualize_model.py
"""

from datetime import datetime
from stable_baselines3 import PPO, SAC
from params import *
from model_funcs import *
from collections import defaultdict

def evaluate_agent(model, eval_env, n_episodes=1200):
    difficulty_levels = 4
    episodes_per_difficulty = n_episodes // difficulty_levels

    success_count = 0
    out_of_env_count = 0
    collision_count = 0
    violation_count = 0
    total_reward = 0

    # Store per-difficulty stats
    per_difficulty = defaultdict(lambda: {
        "success": 0,
        "out_of_env": 0,
        "collision": 0,
        "violation": 0,
        "reward": 0,
    })

    for i in range(n_episodes):
        current_difficulty = i // episodes_per_difficulty
        eval_env.set_difficulty(current_difficulty)

        obs, _ = eval_env.reset()
        terminated = False
        truncated = False
        ep_reward = 0
        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += reward

        # Total stats
        total_reward += ep_reward
        success_count += int(info.get("success", False))
        out_of_env_count += int(info.get("out_of_env", False))
        collision_count += int(info.get("collision", False))
        violation_count += int(info.get("rule_violation", False))

        # Per-difficulty stats
        d = per_difficulty[current_difficulty]
        d["reward"] += ep_reward
        d["success"] += int(info.get("success", False))
        d["out_of_env"] += int(info.get("out_of_env", False))
        d["collision"] += int(info.get("collision", False))
        d["violation"] += int(info.get("rule_violation", False))

    # Compute per-difficulty metrics
    per_difficulty_results = {}
    for diff_level, stats in per_difficulty.items():
        per_difficulty_results[diff_level] = {
            "success_rate": stats["success"] / episodes_per_difficulty,
            "out_of_env_rate": stats["out_of_env"] / episodes_per_difficulty,
            "collision_rate": stats["collision"] / episodes_per_difficulty,
            "rule_violation_rate": stats["violation"] / episodes_per_difficulty,
            "avg_reward": stats["reward"] / episodes_per_difficulty,
        }

    # Total metrics
    total_results = {
        "success_rate": success_count / n_episodes,
        "out_of_env_rate": out_of_env_count / n_episodes,
        "collision_rate": collision_count / n_episodes,
        "rule_violation_rate": violation_count / n_episodes,
        "avg_reward": total_reward / n_episodes,
        "per_difficulty": per_difficulty_results
    }

    return total_results

def save_results_to_txt(results, filepath="RL_training/results_multi_task/eval_results.txt"):
    
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    with open(filepath, "w", encoding="utf-8") as f:
        # --------------------------------------------------
        # Header
        # --------------------------------------------------
        f.write(f"=== Evaluation Results ({timestamp}) ===\n")

        # Overall numbers
        f.write(f"Success Rate:         {results['success_rate']:.2f}\n")
        f.write(f"Out of Env Rate:      {results['out_of_env_rate']:.2f}\n")
        f.write(f"Collision Rate:       {results['collision_rate']:.2f}\n")
        f.write(f"Rule Violation Rate:  {results['rule_violation_rate']:.2f}\n")
        f.write(f"Average Reward:       {results['avg_reward']:.2f}\n\n")

        # Per-difficulty numbers
        f.write("=== Per-Difficulty Breakdown ===\n")
        for diff, stats in results["per_difficulty"].items():
            f.write(f"Difficulty {diff}:\n")
            f.write(f"  Success Rate:         {stats['success_rate']:.2f}\n")
            f.write(f"  Out of Env Rate:      {stats['out_of_env_rate']:.2f}\n")
            f.write(f"  Collision Rate:       {stats['collision_rate']:.2f}\n")
            f.write(f"  Rule Violation Rate:  {stats['rule_violation_rate']:.2f}\n")
            f.write(f"  Average Reward:       {stats['avg_reward']:.2f}\n\n")

    print(f"Saved evaluation results to {os.path.abspath(filepath)}")

model = SAC.load("RL_training/sac_model.zip")
env = create_env()
results = evaluate_agent(model, env)
save_results_to_txt(results)