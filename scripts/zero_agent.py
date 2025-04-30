# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
# parser.add_argument("--enable_cameras", action="store_true" , default=False, help="Enable camera.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to simulate.")
# parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Iiwa-IK-Rel-v0", help="Name of the task.")
parser.add_argument("--task", type=str, default="Isaac-Full-Obj-PC-Lift-Iiwa-IK-Rel-v0", help="Name of the task.")


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
    default="1,8",#None,
    help="A comma separated list of identifiers to be taken from the dataset",
)

# parser.add_argument("--device", type=str, default="cpu", help="Name of the device.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)

simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import IsaacGraspEnv.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from IsaacGraspEnv.dataset_managers import load_object_dataset, preprocessing_point_cloud_loading


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    # env_cfg = parse_env_cfg(
    #     args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    # )
    # create environment
    env_cfg = parse_env_cfg(
        args_cli.task, device="cpu", num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
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
    
    env_cfg.observations.policy.point_cloud.params["path_to_point_clouds"] = preprocessing_point_cloud_loading(args_cli.dataset_path, "point_cloud_colorless.ply", dt_models_filter)
    env_cfg.observations.policy.point_cloud.params["scale"] = 0.001
    env_cfg.observations.policy.point_cloud.params["num_pc"] = 500
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    init_obs, __ = env.reset()
    # simulate environment
    
    while simulation_app.is_running():
        # run everything in inference mode
        i = 0
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # apply actions
            # actions[:,:6] = init_obs["policy"][:,:6]
            # actions[:,6:] = torch.ones((env.action_space.shape[0],env.action_space.shape[1]-6), device=env.unwrapped.device) * 1
            # actions[1,6:] = torch.ones((1,env.action_space.shape[1]-6), device=env.unwrapped.device) * (-1)
            obs, _, done, terminated, info = env.step(actions) # type: ignore
            # if done or terminated:
            # i = 0

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
