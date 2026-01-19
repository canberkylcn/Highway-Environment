import os
from stable_baselines3.common.callbacks import BaseCallback

class SaveHalfwayCallback(BaseCallback):
    """
    A custom callback to save the model exactly at 50% of training.
    """
    def __init__(self, save_path: str, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.halfway_point = total_timesteps // 2
        self.has_saved = False

    def _on_step(self) -> bool:
        # Check if we crossed the halfway mark
        # Num_timesteps bazen tam sayıya denk gelmez, o yüzden >= kullanıyoruz
        if not self.has_saved and self.num_timesteps >= self.halfway_point:
            
            print(f"\n🚀 Reached 50% progress! Saving half-trained model to {self.save_path}...")
            
            # Klasörün var olduğundan emin ol
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            
            self.model.save(self.save_path)
            self.has_saved = True
            
        return True