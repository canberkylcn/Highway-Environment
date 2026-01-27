import argparse
import gymnasium as gym
import highway_env
import os
import sys

sys.path.append(os.getcwd())

from src.utils.file_handler import load_config
from src.wrappers.highway_wrapper import HighwayConfigWrapper
from src.agents.sb3_manager import SB3AgentManager
from src.utils.video_utils import record_agent_run

def get_args():
    parser = argparse.ArgumentParser(description="Train an agent on Highway-Env.")
    parser.add_argument('--env', type=str, default='highway', help='Config file name (e.g. highway, merge)')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'visualize'])
    return parser.parse_args()

def main():
    args = get_args()
    config_path = f"config/{args.env}.yaml"
    
    print(f"--- Running Mode: {args.mode.upper()} ---")

    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"CRITICAL ERROR: Config file not found! {e}")
        return
    
    env_name = config['env_id'] 

    if args.mode == 'train':
        agent_manager = SB3AgentManager(config=config, env=None, mode='train')
        agent_manager.train()
        agent_manager.save_fully_trained() 

    else:
        base_env = gym.make(config['env_id'], render_mode='rgb_array')
        env = HighwayConfigWrapper(base_env, config['env_params'])
        agent_manager = SB3AgentManager(config=config, env=env, mode='test')

        if args.mode == 'test':
            final_path = f"models/{env_name}/fully_trained_{env_name}_model.zip"
            try:
                agent_manager.load(final_path)
                obs, info = env.reset()
                done = False
                truncated = False
                while not (done or truncated):
                    action, _ = agent_manager.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(action)
                    env.render()
                print("Test finished.")
            except Exception as e:
                print(f"Error loading model: {e}")

        elif args.mode == 'visualize':
            print(f"Generating progression videos for {env_name}...")
            video_folder = f"logs/videos/{env_name}"
            os.makedirs(video_folder, exist_ok=True)

            untrained_path = f"models/{env_name}/untrained_{env_name}_model.zip"
            print(f"1. Recording Untrained ({untrained_path})...")
            
            if os.path.exists(untrained_path):
                agent_manager.load(untrained_path)
                record_agent_run(env, model=agent_manager.model, video_folder=video_folder, name_prefix="1_untrained")
            else:
                print("   File not found! Using random.")
                record_agent_run(env, model=None, video_folder=video_folder, name_prefix="1_untrained_random")

            half_path = f"models/{env_name}/half_trained_{env_name}_model.zip"
            print(f"2. Recording Half-Trained ({half_path})...")
            
            if os.path.exists(half_path):
                agent_manager.load(half_path)
                record_agent_run(env, model=agent_manager.model, video_folder=video_folder, name_prefix="2_half_trained")
            else:
                print("   Half-trained file not found.")
            
            full_path = f"models/{env_name}/fully_trained_{env_name}_model.zip"
            print(f"3. Recording Fully Trained ({full_path})...")
            
            if os.path.exists(full_path):
                agent_manager.load(full_path)
                record_agent_run(env, model=agent_manager.model, video_folder=video_folder, name_prefix="3_fully_trained")
            else:
                print("   Fully-trained file not found.")

            env.close()

if __name__ == "__main__":
    main()