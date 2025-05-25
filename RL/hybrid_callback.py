import os
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

from params import *
from model_funcs import *

class HybridCallback(BaseCallback):

    """
    A custom callback function for SB3 PPO training.
    """

    def __init__(
        self,
        n_eval_episodes: int,
        eval_freq: int,
        save_path: str,
        log_path: str,
        render: bool = False,
        verbose: int = 0):
        
        """
        Parameters
        ----------
        :param n_eval_episodes: Number of episodes for evaluation
        :param eval_freq: Frequency (in timesteps) for evaluating the model
        :param save_freq: Frequency (in timesteps) for saving the model
        :param save_path: Directory where the model checkpoints will be saved 
        :param log_path: Directory where the evaluations will be saved
        :param render: Whether or not to render the environment during evaluation
        :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
        """

        super().__init__(verbose)
    
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.model_save_folder = os.path.join(save_path, "models")
        self.render = render
        self.verbose = verbose
        self.best_reward_per_timestep = -float('inf')
        self.writer = SummaryWriter(log_dir=log_path)
        self.eval_metrics = {}
        self.eval_env = create_env()

        os.makedirs(self.model_save_folder, exist_ok=True)  # Creates the folder if it doesn't exist


    def _on_step(self) -> bool:
        """
        Runs after every training timestep.

        Functions:
        ----------
        * Evaluates model after eval_freq timesteps
        * Saves the model after save_freq timesteps
        * Saves the best model during training
        """

        # Evaluate model periodically
        if self.num_timesteps == self.eval_freq * 10:
            
            self.eval_env.reset()
            self.eval_metrics = custom_evaluate_model(
                eval_env=self.eval_env,
                model=self.model,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
                render=self.render
            )

            # Save best model
            avg_reward = np.average(self.eval_metrics["reward"])
            avg_timesteps = np.average(self.eval_metrics["timesteps"])
            if avg_reward / avg_timesteps > self.best_reward_per_timestep:
                self.model.save(os.path.join(self.model_save_folder, "best.zip"))
            
            # Log eval metrics to tensorboard 
            for metric_name, metric_list in self.eval_metrics.items():
                self.writer.add_scalar(f"eval/mean_{metric_name}", 
                                       np.average(metric_list), self.num_timesteps)

        # # Save model periodically 
        # if self.num_timesteps % self.save_freq == 0:
        #     self.model.save(os.path.join(self.model_save_folder, f"{self.num_timesteps//1000}k.zip"))

        return True  # Return false if training aborted early

    def _on_rollout_end(self):
        if bool(self.eval_metrics):
            self.logger.record("eval/mean_reward", np.average(self.eval_metrics["reward"]))
            self.logger.record(f"eval/mean_ep_len (max:{(self.eval_env.end_time/self.eval_env.time_step):.0f})", np.average(self.eval_metrics["timesteps"]))
            self.logger.record("eval/mean_no_of_collisions ", np.average(self.eval_metrics["no_of_collisions"]))
            self.logger.record("eval/mean_no_of_goals ", np.average(self.eval_metrics["no_of_goals"]))
            self.logger.record("eval/mean_no_of_truncations ", np.average(self.eval_metrics["no_of_truncations"]))

    def _on_training_end(self) -> None:
        """
        Finalises the callback by closing the TensorBoard writer.
        """
        self.writer.close()  # Close the writer at the end of training
