import gymnasium as gym
from typing import Dict, Any

class HighwayConfigWrapper(gym.Wrapper):
    """
    A wrapper that applies a specific configuration dictionary to the 
    highway-env environment upon initialization.
    """

    def __init__(self, env: gym.Env, config_params: Dict[str, Any]):
        super().__init__(env)
        self.config_params = config_params
        self._apply_config()

    def _apply_config(self):
        """
        Injects the configuration into the unwrapped environment and updates spaces.
        """
        # 1. Configure the internal environment
        self.env.unwrapped.configure(self.config_params)
        
        # 2. Force a reset to apply changes within highway-env logic
        self.env.reset()

        # 3. CRITICAL FIX: Synchronize the Wrapper's spaces with the Env's new spaces
        # Without this, SB3 sees the old (default) shape (5,5) instead of new (5,7)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, **kwargs):
        """
        Resets the environment.
        """
        return self.env.reset(**kwargs)