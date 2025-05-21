from stable_baselines3 import PPO
from RL.params import *
from RL.env import MyEnv

def create_env(rec=False, vid_name="Current", sim_status=True):
    return MyEnv(
        agent_start_pos_longlat,
        goal_pos_longlat,
        heading,
        max_velocity_knots,
        cruising_speed_knots,
        max_acc_ms2,
        max_yaw_rate_degs,
        detection_radius,
        min_obs_detection_radius,
        screen_height,
        screen_width,
        margins,
        ops_bubble_multiplier,
        grid_number,
        decision_rate,
        display_rate,
        colours_dict,
        max_obstacles,
        safety_radius_dict,
        rewards_weights_dict,
        entity_size,
        proximity_to_goal,

        obstacle_motion_type,
        max_spawned_obs=no_of_generated_obs,
        
        simulation_status=sim_status,
        record=rec,
        video_name=vid_name,
    )

def custom_evaluate_model(
    eval_env: MyEnv,
    model: PPO,
    n_eval_episodes: int = 1,
    deterministic: bool = True,
    render: bool = False,
):
    """
    Evaluates model performance.

    :param eval_env: Env for model evaluation
    :param model: SB3 PPO model to be evaluated
    :param n_eval_episodes: Number of episodes for evaluation of the model. Avg will be used for metrics
    :param render: Renders env durign evaluation
    
    :return: Evaluation metrics dictionary
    """

    metrics = {"reward":[],"no_of_goals":[],"no_of_collisions":[],"timesteps":[],"no_of_truncations":[]}

    for ep in range(n_eval_episodes):

        ep_metrics = {"reward": 0,"no_of_goals": 0,"no_of_collisions": 0,"timesteps":0, "no_of_truncations":0}

        obs, _ = eval_env.reset()
        terminated = truncated = False

        # Run model in env
        while not (terminated or truncated):

            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _  = eval_env.step(action)

            if render:
                eval_env.render()

            ep_metrics["no_of_goals"] = eval_env.rewards_log["goal_reward"] // eval_env.reward_weights_dict["goal_reward_weightage"]
            ep_metrics["no_of_collisions"] = eval_env.rewards_log["collision_penalty"] // eval_env.reward_weights_dict["obs_collision_penalty_weightage"]
            ep_metrics["reward"] += reward
            ep_metrics["timesteps"] += 1
            ep_metrics["no_of_truncations"] = 1 if truncated else 0

        eval_env.close()
        
        metrics["reward"].append(ep_metrics["reward"])
        metrics["no_of_goals"].append(ep_metrics["no_of_goals"])
        metrics["no_of_collisions"].append(ep_metrics["no_of_collisions"])
        metrics["timesteps"].append(ep_metrics["timesteps"])
        metrics["no_of_truncations"].append(ep_metrics["no_of_truncations"])

    return metrics