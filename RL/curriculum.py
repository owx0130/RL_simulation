from stable_baselines3.common.callbacks import BaseCallback

# Determine the cumulative boundary for each difficulty
# Difficulty 0 - no obstacles
# Difficulty 1 - static obstacles scattered randomly
# Difficulty 2 - head-on situation, agent must avoid while abiding COLREGS
# Difficulty 3 - crossing situation, agent must avoid while abiding COLREGS
# Difficulty 4 - overtaking situation, agent must avoid while abiding COLREGS
# Difficulty 5 - obstacle can be any of the 3 COLREGs situations
# Difficulty 6 - multiple moving/static obstacles
# Each constant here indicates at which proportion of the total timesteps their represented difficulty will end
DIFFICULTY_0 = 0.1
DIFFICULTY_1 = 0.1
DIFFICULTY_2 = 0.1
DIFFICULTY_3 = 0.1
DIFFICULTY_4 = 0.35
DIFFICULTY_5 = 0.6

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
        elif progress < DIFFICULTY_4:
            difficulty = 4
        elif progress < DIFFICULTY_5:
            difficulty = 5
        else:
            difficulty = 6
        
        return difficulty