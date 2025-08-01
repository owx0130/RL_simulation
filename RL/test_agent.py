from datetime import datetime
from stable_baselines3 import SAC
from params import *
from model_funcs import *
from collections import defaultdict

RESULTS_SAVE_PATH = "RL_training/results_multi_task/eval_results.txt"
MODEL_PATH = "RL_training/sac_model.zip"

def evaluate_agent(model, eval_env, n_episodes=6000):
    # Evaluate on goalfinding, static obstacles, 3 COLREGs situations, and multiple obstacles
    evaluated_difficulty_levels = [0, 1, 2, 3, 4, 6]
    episodes_per_difficulty = n_episodes // len(evaluated_difficulty_levels)

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
        current_difficulty = evaluated_difficulty_levels[i // episodes_per_difficulty]
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

def save_results_to_txt(results, filepath=RESULTS_SAVE_PATH):
    
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    with open(output_path, "w", encoding="utf-8") as f:
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

# Load and evaluate model
model = SAC.load(MODEL_PATH)
env = create_env()
results = evaluate_agent(model, env)

# Save the evaluation results to a text file
save_results_to_txt(results)