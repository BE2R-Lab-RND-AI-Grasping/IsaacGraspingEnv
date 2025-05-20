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
    "--num_envs", type=int, default=5, help="Number of environments to simulate."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Lift-Cube-Iiwa-IK-Rel-v0",
    help="Name of the task.",
)

parser.add_argument(
    "--dataset_path",
    type=str,
    default="/home/yefim-home/Documents/work/IsaacGraspingEnv/source/IsaacGraspEnv/IsaacGraspEnv/assets/data/HANDEL/screwdrivers",#None,
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
    default="4",#None,
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

from IsaacGraspEnv.dataset_managers import load_object_dataset


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
    

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            actions = (
                2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            )
            # apply actions
            env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
