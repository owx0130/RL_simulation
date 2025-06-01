from stable_baselines3.common.callbacks import BaseCallback

# Determine the cumulative boundary for each difficulty
# Difficulty 0 - no obstacles
# Difficulty 1 - static obstacles scattered randomly
# Difficulty 2 - more static obstacles scattered randomly
# Difficulty 3 - head on obstacles, agent must avoid while abiding COLREGS
DIFFICULTY_0 = 0.15
DIFFICULTY_1 = 0.25
DIFFICULTY_2 = 0.4
DIFFICULTY_3 = 0.7

class CurriculumCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps

    def _on_step(self):
        progress = self.num_timesteps / self.total_timesteps
        new_difficulty = self._calculate_difficulty(progress)

        if hasattr(self.training_env, "envs"):
            for env in self.training_env.envs:
                if env.unwrapped.difficulty == new_difficulty:
                    break
                else:
                    env.unwrapped.set_difficulty(new_difficulty)
                    print(f"[CurriculumCallback] Updated difficulty to {new_difficulty} at step {self.num_timesteps}")

        else:
            self.training_env.set_difficulty(new_difficulty)
            print(f"[CurriculumCallback] Updated difficulty to {new_difficulty} at step {self.num_timesteps}")
        
        return True

    def _calculate_difficulty(self, progress):
        if progress < DIFFICULTY_0:
            difficulty = 0
        elif progress < DIFFICULTY_1:
            difficulty = 1
        elif progress < DIFFICULTY_2:
            difficulty = 2
        elif progress < DIFFICULTY_3:
            difficulty = 3
        else:
            difficulty = 4
        
        return difficulty