import os
from typing import Dict, Any, Optional, Callable
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import HerReplayBuffer 

import gymnasium as gym
from src.wrappers.highway_wrapper import HighwayConfigWrapper
from src.utils.callbacks import SaveHalfwayCallback

def linear_schedule(initial_value: float) -> Callable[[float], float]:
   
    def func(progress_remaining: float) -> float:
        
        return progress_remaining * initial_value
    return func

class SB3AgentManager:
    def __init__(self, config: Dict[str, Any], env: Optional[gym.Env] = None, mode: str = 'train'):
        self.config = config
        self.agent_params = config['agent_params']
        self.env_name = config['env_id']
        
        if mode == 'train':
            print(f"🚀 Initializing TRUE Multiprocessing (4 CPUs) for {self.env_name}...")
            self.env = make_vec_env(
                env_id=self.env_name,
                n_envs=8,
                wrapper_class=HighwayConfigWrapper,
                wrapper_kwargs={"config_params": config['env_params']},
                vec_env_cls=SubprocVecEnv
            )
        else:
            self.env = env

        self.model = self._create_model()

    def _create_model(self) -> BaseAlgorithm:
        algo_name = self.agent_params['algorithm'].upper()
        model_kwargs = self.agent_params.get('model_params', {}).copy() # Kopyasını al ki orijinali bozulmasın
        
        if 'learning_rate' in model_kwargs and isinstance(model_kwargs['learning_rate'], (float, int)):
            model_kwargs['learning_rate'] = linear_schedule(model_kwargs['learning_rate'])

        tb_log = os.path.join(self.agent_params.get('tensorboard_log', "logs/"), self.env_name)
        
        if algo_name == "SAC":
            # SAC her zaman MultiInputPolicy kullanmalı (Goal için)
            policy_type = "MultiInputPolicy"
            
            if model_kwargs.get("replay_buffer_class") == "HerReplayBuffer":
                model_kwargs["replay_buffer_class"] = HerReplayBuffer
            
            return SAC(policy_type, self.env, tensorboard_log=tb_log, **model_kwargs)

        elif algo_name == "DQN":
            return DQN("MlpPolicy", self.env, tensorboard_log=tb_log, device='auto', **model_kwargs)
        elif algo_name == "PPO":
            # Roundabout ve Highway için MlpPolicy devam
            return PPO("MlpPolicy", self.env, tensorboard_log=tb_log, device='auto', **model_kwargs)
        else:
            raise ValueError(f"Algorithm {algo_name} not supported yet.")

    def train(self):
        timesteps = self.agent_params['total_timesteps']
        
        untrained_path = f"./models/{self.env_name}/untrained_{self.env_name}_model"
        print(f"Saving untrained model to {untrained_path}...")
        os.makedirs(os.path.dirname(untrained_path), exist_ok=True)
        self.model.save(untrained_path)
        
        callbacks = []

        half_path = f"./models/{self.env_name}/half_trained_{self.env_name}_model"
        half_callback = SaveHalfwayCallback(save_path=half_path, total_timesteps=timesteps)
        callbacks.append(half_callback)

        freq = self.agent_params.get('checkpoint_freq', 0)
        if freq > 0:
            save_freq = max(freq // 4, 1)
            checkpoint_path = f"./models/{self.env_name}/checkpoints/"
            ckpt_callback = CheckpointCallback(
                save_freq=save_freq,
                save_path=checkpoint_path,
                name_prefix='checkpoint'
            )
            callbacks.append(ckpt_callback)

        print(f"Starting training for {timesteps} timesteps on {self.env_name}...")
        self.model.learn(total_timesteps=timesteps, callback=CallbackList(callbacks))
        print("Training finished.")

    def save_fully_trained(self):
        final_path = f"./models/{self.env_name}/fully_trained_{self.env_name}_model"
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        self.model.save(final_path)
        print(f"Fully trained model saved at: {final_path}.zip")

    def load(self, path: str):
        if path.endswith(".zip"):
            path = path[:-4]
        
        algo_name = self.agent_params['algorithm'].upper()
        if algo_name == "DQN":
            self.model = DQN.load(path, env=self.env)
        elif algo_name == "PPO":
            self.model = PPO.load(path, env=self.env)
        elif algo_name == "SAC":
            self.model = SAC.load(path, env=self.env)
        print(f"Model loaded from: {path}")