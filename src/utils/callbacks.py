import os
from stable_baselines3.common.callbacks import BaseCallback

class SaveHalfwayCallback(BaseCallback):
   
    def __init__(self, save_path: str, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.halfway_point = total_timesteps // 2
        self.has_saved = False

    def _on_step(self) -> bool:
        
        if not self.has_saved and self.num_timesteps >= self.halfway_point:
            
            print(f"\n🚀 Reached 50% progress! Saving half-trained model to {self.save_path}...")
            
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            
            self.model.save(self.save_path)
            self.has_saved = True
            
        return True