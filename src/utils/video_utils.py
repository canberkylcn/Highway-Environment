import gymnasium as gym
from stable_baselines3.common.base_class import BaseAlgorithm
from gymnasium.wrappers import RecordVideo
import os

def record_agent_run(env: gym.Env, model: BaseAlgorithm, video_folder: str, name_prefix: str, steps: int = 500):
    env = RecordVideo(
        env, 
        video_folder=video_folder, 
        name_prefix=name_prefix,
        episode_trigger=lambda x: True
    )

    obs, info = env.reset()
    done = False
    
    for _ in range(steps):
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if done:
            break
    
    env.close()
    print(f"Video saved to {video_folder}/{name_prefix}-episode-0.mp4")