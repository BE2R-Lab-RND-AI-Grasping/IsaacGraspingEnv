# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Vectors-Lift-Iiwa-IK-Rel-v0", #"Isaac-Full-Obj-PC-Lift-Iiwa-IK-Rel-v0",
    help="Name of the task.",
)

parser.add_argument(
    "--dataset_path",
    type=str,
    default="/home/yefim-home/Documents/work/IsaacGraspingEnv/source/IsaacGraspEnv/IsaacGraspEnv/assets/data/HANDEL/power_drills",#None,
    help="Absolute path to dataset. Dataset directory must have folders with models.",
)
parser.add_argument(
    "--usd_file_name",
    type=str,
    default="object.usd",
    help="The name of the USD file in the folder",
)
parser.add_argument(
    "--model_filter",
    type=str,
    default="1",#None,
    help="A comma separated list of identifiers to be taken from the dataset",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch


import IsaacGraspEnv.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from IsaacGraspEnv.dataset_managers import load_object_dataset, preprocessing_point_cloud_loading
from IsaacGraspEnv.debug_function.reward_analyze import RewardAnalyzer

def fabric_func_unpack_obs(env, name_obs_vector):
    # setup function for unpack observation
    id_obs_vector = env.observation_manager.active_terms["policy"].index(name_obs_vector)
    size_obs = env.observation_manager.group_obs_term_dim["policy"][id_obs_vector][0]
    index_obs_vector = sum([env.observation_manager.group_obs_term_dim["policy"][i][0] for i in range(id_obs_vector)])
    unpack_vectors4obs = lambda env: env.observation_manager._obs_buffer["policy"][:,index_obs_vector:index_obs_vector+size_obs]
    
    return unpack_vectors4obs


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        # args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        args_cli.task,
        device="cpu",
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.model_filter:
        dt_models_filter = args_cli.model_filter.replace(" ", "").split(",")
    else:
        dt_models_filter = args_cli.model_filter
        

    env_cfg.scene.object.rigid_objects =  load_object_dataset(
        args_cli.dataset_path,
        args_cli.usd_file_name,
        dt_models_filter
    )
    
    if  args_cli.task ==  "Isaac-Full-Obj-PC-Lift-Iiwa-IK-Rel-v0":
        env_cfg.observations.policy.point_cloud.params["path_to_point_clouds"] = preprocessing_point_cloud_loading(args_cli.dataset_path, "point_cloud_colorless.ply", dt_models_filter)
        env_cfg.observations.policy.point_cloud.params["scale"] = 0.01
        env_cfg.observations.policy.point_cloud.params["num_pc"] = 500
    # elif args_cli.task ==  "Isaac-Vectors-Lift-Iiwa-IK-Rel-v0": 
        # env_cfg.observations.policy.vectors.params["path_to_point_clouds"] = preprocessing_point_cloud_loading(args_cli.dataset_path, "point_cloud_colorless.ply", dt_models_filter)
        # env_cfg.observations.policy.vectors.params["scale"] = 0.01
        # env_cfg.observations.policy.vectors.params["num_pc"] = 500

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    reward_log = RewardAnalyzer(env.env.reward_manager, env.env.max_episode_length, env.env.num_envs)
    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    
    # func for unpuck ee_frame
    # get_ee_frame = fabric_func_unpack_obs(env, "ee_frame")
    
    # simulate environment
    m = 0
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            actions = (
                10 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            )
            # apply actions
            reward_log.step_update()
            obs, rew, terminated, truncated, info = env.step(actions)
            # ee_frame = get_ee_frame(env)
            
            # mean_ee_frame = ee_frame.mean(dim=0)
            
            for i in range(env.env.num_envs):
                if terminated[i] or truncated[i]:
                    reward_log.episode_update(i)
                    if truncated[i]:
                        m += 1
            
            
            if m > 1000:
                break
            

    # close the simulator
    env.close()
    reward_log.plot()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
