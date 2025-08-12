# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with actions from generated poses using DexGraspNet"""
"""Example of Using 
python scripts/generated_grasp_test.py --task Isaac-Testing-BH-Manager-v0 --dataset_path /home/rahaf/Lab/nirsii/IsaacGraspingEnv/dataset_usd/locking_pliers/ --model_filter "0" --object_kinematic_enabled --enable_cameras --device cuda --generated_poses_path assets/poses/pliers_0.npy 
"""

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
    "--task",
    type=str,
    default="Isaac-Testing-BH-Manager-v0",
    help="Name of the task.",
)

parser.add_argument(
    "--dataset_path",
    type=str,
    default=None,
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
    default=None,
    help="A comma separated list of identifiers to be taken from the dataset",
)
parser.add_argument(
    "--object_kinematic_enabled",  # True or False
    action="store_true",
    default=False,
    help="Enable kinematic for the object.",
)
parser.add_argument(
    "--generated_poses_path",
    type=str,
    required=True,
    help="Path to the npy file with generated poses .",
)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# check generated poses path exists
import os
import numpy as np
import matplotlib.pyplot as plt
import transforms3d.euler as euler
from transforms3d import affines

if not os.path.exists(args_cli.generated_poses_path):
    raise FileNotFoundError(
        f"Generated poses path {args_cli.generated_poses_path} does not exist."
    )

generated_poses = np.load(args_cli.generated_poses_path, allow_pickle=True)
num_poses = len(generated_poses)
CARTESIAN_ACTION_KEYS = [
    "WRJTz",
    "WRJTy",
    "WRJTx",
    "WRJRx",
    "WRJRy",
    "WRJRz",
]

GRIPPER_ACTION_KEYS = [
    "wam_bhand_finger_3_med_joint",
    "wam_bhand_finger_3_dist_joint",
    "wam_bhand_finger_1_prox_joint",
    "wam_bhand_finger_1_med_joint",
    "wam_bhand_finger_1_dist_joint",
    "wam_bhand_finger_2_prox_joint",
    "wam_bhand_finger_2_med_joint",
    "wam_bhand_finger_2_dist_joint",
]


def get_cartesian_action(pose):
    return [pose[key] for key in CARTESIAN_ACTION_KEYS]


def get_gripper_action(pose):
    return [pose[key] for key in GRIPPER_ACTION_KEYS]


for pose_id in range(num_poses):
    print(f"Pose {pose_id}")
    print("CARTESIAN ACTION", get_cartesian_action(generated_poses[pose_id]["qpos"]))
    print("GRIPPER ACTION", get_gripper_action(generated_poses[pose_id]["qpos"]))


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch


import IsaacGraspEnv.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from IsaacGraspEnv.dataset_managers import load_object_dataset
from typing import Optional
import numpy as np


def main():

    generated_poses = np.load(args_cli.generated_poses_path, allow_pickle=True)
    print(generated_poses.shape)
    # create environment configuration
    env_cfg = parse_env_cfg(
        # args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        args_cli.task,
        device="cpu",
        num_envs=num_poses,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.model_filter:
        dt_models_filter = args_cli.model_filter.replace(" ", "").split(",")
    else:
        dt_models_filter = args_cli.model_filter

    env_cfg.scene.object.rigid_objects = load_object_dataset(
        args_cli.dataset_path,
        args_cli.usd_file_name,
        dt_models_filter,
        kinematic_enabled=args_cli.object_kinematic_enabled,
    )

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    reward_log = RewardAnalyzer(
        env.env.reward_manager, env.env.max_episode_length, env.env.num_envs
    )

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    rewards = []
    m = 0
    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = (
                torch.zeros(env.action_space.shape, device=env.unwrapped.device) - 1
            )
            step += 1
            for pose_id in range(num_poses):
                actions[pose_id, : len(CARTESIAN_ACTION_KEYS)] = torch.tensor(
                    get_cartesian_action(generated_poses[pose_id]["qpos"]),
                    device=env.unwrapped.device,
                )
                actions[pose_id, 0] += 0.5
                if step > 100:
                    actions[pose_id, len(CARTESIAN_ACTION_KEYS) :] = torch.tensor(
                        get_gripper_action(generated_poses[pose_id]["qpos"]),
                        device=env.unwrapped.device,
                    )
            reward_log.step_update()
            _, rw, term, trun, _ = env.step(actions)
            rewards.append(rw.clone().detach().cpu().numpy())
            for i in range(env.env.num_envs):
                if term[i] or trun[i]:
                    reward_log.episode_update(i)
                    m += 1
                    step = 0
            if m >= 10:
                reward_log.plot()
                m = 0

    # close the simulator
    env.close()


def plot_rewards(rewards):
    """Plot rewards."""
    num_lines = rewards.shape[1]
    fig = plt.figure(figsize=(10, 5))
    for i in range(num_lines):
        plt.plot(rewards[:, i], label=f"Pose {i}")
    plt.xlabel("Time step")
    plt.ylabel("Reward")
    plt.title("Rewards for each pose")
    plt.legend()
    plt.show()


class RewardAnalyzer:

    def __init__(self, reward_manager, episode_length, num_envs=1) -> None:

        self.reward_manager = reward_manager

        self._env = self.reward_manager._env
        self.term_names = self.reward_manager._term_names
        self._term_cfgs = self.reward_manager._term_cfgs
        self.episode_length = episode_length
        self.num_envs = num_envs

        self.ep_history = {
            name: [[] for __ in range(num_envs)] for name in self.term_names
        }
        self.term_history = {name: [] for name in self.term_names}

    def step_update(self):

        for term_idx, (name, term_cfg) in enumerate(
            zip(self.term_names, self._term_cfgs)
        ):
            step_reward = term_cfg.func(self._env, **term_cfg.params).cpu().numpy()
            for i in range(self.num_envs):

                self.ep_history[name][i].append(step_reward[i])

    def episode_update(self, id_env=0):

        for name in self.term_history.keys():
            if len(self.ep_history[name][id_env]) == self.episode_length:
                self.term_history[name].append(self.ep_history[name][id_env])
            self.ep_history[name][id_env] = []

    def plot(self, episode: Optional[int] = None):

        terms_name = self.term_history.keys()
        n = len(terms_name)  # number of plots
        cols = np.ceil(np.sqrt(n))
        rows = np.ceil(n / cols)

        fig, axes = plt.subplots(int(rows), int(cols), figsize=(22, 15.4))
        axes = axes.flatten()
        term_weights = [cfg.weight for cfg in self._term_cfgs]
        term_weights = term_weights  # + [0.0, 0.0]
        for ax, name, term_weight in zip(axes, terms_name, term_weights):
            if episode:
                data = np.asarray(self.term_history[name][episode]).squeeze()
                ax.plot(np.arange(data.shape[-1]), data, linewidth=2)
            else:
                data = np.asarray(self.term_history[name]).squeeze()
                if len(data.shape) == 1:
                    data = data[np.newaxis, :]
                for i, sample in enumerate(data):
                    ax.plot(
                        np.arange(sample.shape[-1]),
                        sample,
                        alpha=0.6,
                        label=f"Pose {i}",
                    )

            # fig.suptitle(f'Mean reward value {num_samples} samples')
            ax.set_title(f"{name} ep: {episode}; w: {np.round(term_weight, 3)}")
            ax.set_xlabel("Env Step")
            ax.set_ylabel("Value")
            # ax.grid(True)
            ax.legend()

        for i in range(n, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
    simulation_app.close()
