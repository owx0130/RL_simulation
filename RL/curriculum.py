from stable_baselines3.common.callbacks import BaseCallback

class CurriculumCallback(BaseCallback):
    def __init__(self, update_freq, verbose=1):
        super().__init__(verbose)
        self.update_freq = update_freq

    def _on_step(self):
        if self.num_timesteps % self.update_freq == 0:
            new_difficulty = self._calculate_difficulty(self.num_timesteps)

            if hasattr(self.training_env, "envs"):
                for env in self.training_env.envs:
                    env.unwrapped.set_difficulty(new_difficulty)
            else:
                self.training_env.set_difficulty(new_difficulty)

            if self.verbose > 0:
                print(f"[CurriculumCallback] Updated difficulty to {new_difficulty} at step {self.num_timesteps}")
        
        return True

    def _calculate_difficulty(self, step):
        return step // self.update_freq