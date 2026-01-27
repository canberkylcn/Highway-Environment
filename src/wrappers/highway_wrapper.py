import gymnasium as gym
from typing import Dict, Any

class HighwayConfigWrapper(gym.Wrapper):
   

    def __init__(self, env: gym.Env, config_params: Dict[str, Any]):
        super().__init__(env)
        self.config_params = config_params
        self._apply_config()

    def _apply_config(self):
        
        self.env.unwrapped.configure(self.config_params)
        self.env.reset()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, **kwargs):
        
        return self.env.reset(**kwargs)