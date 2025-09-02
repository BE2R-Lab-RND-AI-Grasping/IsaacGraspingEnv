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
    default="Isaac-Vectors-Lift-Iiwa-IK-Rel-v0",
    help="Name of the task.",
)
# append AppLauncher cli args

parser.add_argument(
    "--dataset_path",
    type=str,
    default="/home/yefim-home/Documents/work/IsaacGraspingEnv/source/IsaacGraspEnv/IsaacGraspEnv/assets/data/HANDEL/screwdrivers",
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
    default="1",
    help="A comma separated list of identifiers to be taken from the dataset",
)

parser.add_argument(
    "--grasp_dataset_path",
    type=str,
    default="/home/yefim-home/Documents/work/repo_forks/DexGraspNet/grasp_generation_egorhand_edited_hand/ready_to_work/dataset/DIP-Flex_opened_kinematics",
    help="Absolute path to dataset. Dataset directory must have folders with models.",
)
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import numpy as np


from pathlib import Path
import isaaclab.envs.mdp as mdp  # noqa: F401
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

from IsaacGraspEnv.dataset_managers.dataset_loading import load_object_dataset
import IsaacGraspEnv.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab.utils.math import (
    transform_points,
    subtract_frame_transforms,
    quat_from_euler_xyz,
    combine_frame_transforms,
)

map_old2new_joint_names = {
    "left": "index",
    "right": "pinkie",
    "thumb": "thumb",
    "rotation": "rotation",
    "flexion": "PPflexion",
    "finray_proxy": "DPflexion",
}

translation_names = ["WRJTx", "WRJTy", "WRJTz"]
rot_names = ["WRJRx", "WRJRy", "WRJRz"]


grasp_conf_index = 8


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device="cpu",
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        # args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # env_cfg.terminations.time_out = None
    env_cfg.episode_length_s = 3
    if args_cli.model_filter:
        dt_models_filter = args_cli.model_filter.replace(" ", "").split(",")
    else:
        dt_models_filter = args_cli.model_filter

    # ================SET UP ENV Config for Debugging==========================

    env_cfg.scene.object.rigid_objects = load_object_dataset(
        args_cli.dataset_path, args_cli.usd_file_name, dt_models_filter, True
    )

    env_cfg.actions.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["lbr_iiwa_joint_.*"],
        body_name="lbr_iiwa_link_7",
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        ),
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=[0.0, 0.0, 0.0]
        ),
        # clip={"lbr_iiwa_joint_[1-6]":(-1.0, 1.0)}
    )

    env_cfg.actions.gripper_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "Joint_.*",
        ],
        scale={
            "Joint_thumb_rotation": -1.0,
            "Joint_thumb_abduction": -1.0,
        },
        offset={"Joint_thumb_abduction": np.pi / 2},
    )

    cube_in_focus_pos_range = env_cfg.events.randomize_cubes_in_focus.params[
        "pose_range"
    ]

    cube_in_focus_pos_range["x"] = (0.9, 0.9)
    cube_in_focus_pos_range["y"] = (-0.1, -0.1)
    cube_in_focus_pos_range["z"] = (0.3, 0.3)
    cube_in_focus_pos_range["roll"] = (0.0, 0.0)
    cube_in_focus_pos_range["pitch"] = (1.57, 1.57)
    cube_in_focus_pos_range["yaw"] = (1.57, 1.57)

    env_cfg.scene.ee_frame.target_frames[0].offset.pos = [0.0, 0.0, 0.0]

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.env.sim.set_camera_view([2.5, 1, 1], [0.0, 0.0, 0.0])
    env.env.sim.set_render_mode(env.env.sim.RenderMode.FULL_RENDERING)
    # env.env.sim.set_render_mode(env.env.sim.RenderMode.NO_GUI_OR_RENDERING)
    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    # =======observation=======
    # ee_frame: pos 0:3, quat 3:7; ft_pos: thumb: 7:10 right: 10:13 left: 13:16; normalized joint pos: 16:36; joint vel: 36:56;
    # object pos: 56:59, quat: 59:63; target: 63:70; action: 70:83
    # =========action==========
    # arm_action: delta_pos: 0:3, delta_angle: 3:6; gripper_action: 6:13
    # =====gripper joint=======
    # 0: left_abduction, 1: right_abduction 2: thumb_rotation
    # 3: left_dynamixel, 4: right_dynamixel, 5: thumb_abduction
    # 6: thumb_dynamixel
    obs, __ = env.reset()
    time_step = 0
    time_arr = []
    action_manager = env.get_wrapper_attr("action_manager")
    robot = env.get_wrapper_attr("scene").articulations["robot"]

    l_joint_ordering_env = robot.find_joints(
        env_cfg.actions.gripper_action.joint_names, preserve_order=False
    )[1]

    l_joint_order_new_name = []
    for j_name in l_joint_ordering_env:
        new_j_name = j_name + ""
        for old_key, new_key in map_old2new_joint_names.items():
            if old_key in j_name:
                new_j_name = new_j_name.replace(old_key, new_key, 1)
        l_joint_order_new_name.append(new_j_name)

    path2object_dataset = Path(args_cli.dataset_path)
    path2grasp_dataset = Path(args_cli.grasp_dataset_path)

    object_name = path2object_dataset.name

    path2grasp_dataset = [
        file for file in path2grasp_dataset.glob(object_name[:-1] + "_0.npy")
    ]
    data = []
    for file in path2grasp_dataset:
        with open(file, "rb") as f:
            data += np.load(f, allow_pickle=True).tolist()

    ee_body_id = robot.find_bodies("lbr_iiwa_link_7")[0][0]

    list_obs_term_name = env.observation_manager.active_terms[
        "policy"
    ]  # ["ee_frame", "object_position", "target_object_position"]

    target_gripper_angles = []
    for j_name in l_joint_order_new_name:
        target_gripper_angles.append(
            data[grasp_conf_index].get("qpos", None).get(j_name, 0.0)
        )

    target_gripper_angles = torch.Tensor(target_gripper_angles).to(
        env.get_wrapper_attr("device")
    )

    target_pos_obj = []
    for tran_name in translation_names:
        target_pos_obj.append(
            data[grasp_conf_index].get("qpos", None).get(tran_name, 0.0)
        )
    target_pos_obj = torch.Tensor(target_pos_obj).to(env.get_wrapper_attr("device"))

    target_rpy_obj = []
    for rpy_name in rot_names:
        target_rpy_obj.append(
            data[grasp_conf_index].get("qpos", None).get(rpy_name, 0.0)
        )
    target_rpy_obj = torch.Tensor(target_rpy_obj).to(env.get_wrapper_attr("device"))

    target_quat_obj = quat_from_euler_xyz(
        target_rpy_obj[0], target_rpy_obj[1], target_rpy_obj[2]
    )
    # ==============OBS_UNPACK========================

    dict_obs_term_unpack = {}
    for term in list_obs_term_name:
        id_obs_vector = (
            env.get_wrapper_attr("observation_manager")
            .active_terms["policy"]
            .index(term)
        )
        size_obs = env.get_wrapper_attr("observation_manager").group_obs_term_dim[
            "policy"
        ][id_obs_vector][0]
        index_obs_vector = sum(
            [
                env.get_wrapper_attr("observation_manager").group_obs_term_dim[
                    "policy"
                ][i][0]
                for i in range(id_obs_vector)
            ]
        )
        dict_obs_term_unpack[term] = (
            lambda env, s_id=index_obs_vector, size=size_obs: env.get_wrapper_attr(
                "observation_manager"
            )._obs_buffer["policy"][:, s_id : s_id + size]
        )

    # =====================================

    init_position = []
    succes = []
    rew_arr = []
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():

            # ee_quat_root = env.scene.sensors[
            #     "ee_frame"
            # ].data.target_quat_source.squeeze()
            # ee_pos_root = env.scene.sensors["ee_frame"].data.target_pos_source.squeeze()
            ee_pos_w = robot.data.body_pos_w[:, ee_body_id].squeeze()
            ee_quat_w = robot.data.body_quat_w[:, ee_body_id].squeeze()

            root_pos_w = robot.data.root_pos_w.squeeze()
            root_quat_w = robot.data.root_quat_w.squeeze()

            ee_pos_root, ee_quat_root = subtract_frame_transforms(
                root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
            )

            ee_pos = transform_points(
                dict_obs_term_unpack["ee_frame"](env)[:, 0:3], ee_pos_root, ee_quat_root
            )[
                0
            ]  # Unpack the ee_frame observation
            # Closed Kinematics
            # obj_pos = obs["policy"][0,56:59]  # + torch.Tensor([0.0, 0.0, 0.11]) #4: + torch.Tensor([-0.05, 0.0, 0.1])  #3: + torch.Tensor([-0.05, 0.0, 0.1]) # 2: + torch.Tensor([-0.05, 0.0, 0.1]) #1: + torch.Tensor([-0.1, 0.0, 0.12])
            # target_pos = obs["policy"][0,63:66]
            object_pos_root, object_quat_root = combine_frame_transforms(
                ee_pos_root,
                ee_quat_root,
                dict_obs_term_unpack["object_position"](env).squeeze(0),
                dict_obs_term_unpack["object_quat"](env).squeeze(0),
            )

            target_pos_root, target_quat_root = combine_frame_transforms(
                object_pos_root,
                object_quat_root,
                target_pos_obj,
                target_quat_obj,
            )

            time_arr.append(env.env.sim.current_time)

            # if time_step < 100:
            #     delta_ee_pos = (obj_pos - ee_pos) * 7
            # else:
            #     delta_ee_pos = (target_pos - ee_pos) * 7
            # delta_ee_ang = torch.zeros(3, device=env.unwrapped.device)

            if time_step == 0:
                init_position.append(ee_pos.numpy())

            # if time_step < 50:
            #     # For Closed Kinematics
            #     # gripper_joint = -torch.ones(7, device=env.unwrapped.device)
            #     # gripper_joint[2] = 0
            #     gripper_joint = -torch.ones(10, device=env.unwrapped.device)
            #     gripper_joint[2] = 0
            #     ramp = 1
            # else:
            #     if time_step > 50:
            #         ramp = 1  # 0.01 * (150 - time_step) + 1
            #     else:
            #         ramp = 1  # 0.5 + np.random.normal(0, 0.8) # 0.01 * (time_step - 45) + np.random.normal(0, 0.8)
            #     gripper_joint = -torch.ones(10, device=env.unwrapped.device)
            #     gripper_joint[2] = 0
            #     # gripper_joint[7] = 0
            #     # gripper_joint[3:5] = torch.ones(2, device=env.unwrapped.device) * ramp
            #     # gripper_joint[6] = 1 * ramp
            #     gripper_joint[3:] = (
            #         torch.ones(gripper_joint[3:].shape[0], device=env.unwrapped.device)
            #         * ramp
            #         * 1.0
            #     )

            joint_zeros = torch.zeros(
                action_manager.action_term_dim[
                    action_manager.active_terms.index("gripper_action")
                ],
                device=env.get_wrapper_attr("device"),
            )

            actions = torch.cat(
                [target_pos_root, target_quat_root, joint_zeros]
            ).unsqueeze(0)
            # actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # apply actions
            obs, rew, terminated, truncated, info = env.step(actions)
            time_step += 1
            rew_arr.append(rew)
            # print(time_step, np.round(env.env.sim.current_time, 2), np.round(ramp,3))
            if truncated or terminated:
                if object_pos_root.numpy()[2] > 0.2:
                    succes.append(1)
                else:
                    succes.append(0)
                time_step = 0
                # break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
