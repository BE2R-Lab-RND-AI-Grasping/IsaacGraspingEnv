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
)  # Isaac-Vectors-Lift-Iiwa-IK-Rel-v0
# parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Iiwa-IK-Rel-v0", help="Name of the task.")
# append AppLauncher cli args

parser.add_argument(
    "--dataset_path",
    type=str,
    default="/home/yefim-home/Documents/work/IsaacGraspingEnv/source/IsaacGraspEnv/IsaacGraspEnv/assets/data/HANDEL/power_drills",  # "wrenches",#None,
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
    default="1",  # None,
    help="A comma separated list of identifiers to be taken from the dataset",
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

from IsaacGraspEnv.dataset_managers.dataset_loading import load_object_dataset
import IsaacGraspEnv.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from isaaclab.utils.math import (
    subtract_frame_transforms,
    transform_points,
    unproject_depth,
)

from IsaacGraspEnv.debug_function.reward_analyze import RewardAnalyzer

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
    # env_cfg.episode_length_s = 6
    if args_cli.model_filter:
        dt_models_filter = args_cli.model_filter.replace(" ", "").split(",")
    else:
        dt_models_filter = args_cli.model_filter

    env_cfg.scene.object.rigid_objects = load_object_dataset(
        args_cli.dataset_path, args_cli.usd_file_name, dt_models_filter
    )
    #         # Listens to the required transforms
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"
    env_cfg.scene.target_ft_obj = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/lbr_iiwa_link_0",
        debug_vis=True,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Object_model_1",
                name="target_model_4",
                offset=OffsetCfg(
                    pos=[-0.05, 0.0, 0.0],
                ),
            ),
        ],
    )

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.env.sim.set_camera_view([2.5, 1, 1], [0.0, 0.0, 0.0])
    env.env.sim.set_render_mode(env.env.sim.RenderMode.FULL_RENDERING)

    env.scene.articulations["robot"].root_physx_view.prim_paths

    reward_log = RewardAnalyzer(env.env.reward_manager, env.env.max_episode_length) # Initialize the reward analyzer
    # env.env.sim.set_render_mode(env.env.sim.RenderMode.NO_GUI_OR_RENDERING)
    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    # =======observation=======
    # Closed Kinematics
    # ee_frame: pos 0:3, quat 3:7; ft_pos: thumb: 7:10 right: 10:13 left: 13:16; normalized joint pos: 16:33; joint vel: 33:50;
    # object pos: 50:53, quat: 53:57; target: 57:60; action: 60:76
    # Open Kinematics DP-Flex
    # ee_frame: pos 0:3, quat 3:7; ft_pos: thumb: 7:10 right: 10:13 left: 13:16; normalized joint pos: 16:39; joint vel: 39:62;
    # object pos: 62:65, quat: 65:69; target: 69:72; action: 72:88
    # =========action==========
    # arm_action: delta_pos: 0:3, delta_angle: 3:6; gripper_action: 6:13
    # =====gripper joint=======
    # 0: left_abduction, 1: right_abduction 2: thumb_rotation
    # 3: left_dynamixel, 4: right_dynamixel, 5: thumb_abduction
    # 6: thumb_dynamixel
    obs, __ = env.reset()
    time_step = 0
    time_arr = []

    list_obs_term_name = env.observation_manager.active_terms["policy"] #["ee_frame", "object_position", "target_object_position"]

    dict_obs_term_unpack = {}
    for term in list_obs_term_name:
        id_obs_vector = env.observation_manager.active_terms["policy"].index(term)
        size_obs = env.observation_manager.group_obs_term_dim["policy"][id_obs_vector][0]
        index_obs_vector = sum([env.observation_manager.group_obs_term_dim["policy"][i][0] for i in range(id_obs_vector)])
        dict_obs_term_unpack[term] = lambda env, s_id=index_obs_vector, size=size_obs: env.observation_manager._obs_buffer["policy"][:,s_id:s_id+size]

    init_position = []
    succes = []
    effort_limits = {
        act_name: env.env.scene.articulations["robot"]
        .actuators[act_name]
        .effort_limit.tolist()[0]
        for act_name in env.env.scene.articulations["robot"].actuators.keys()
    }
    computed_efforts_act = {
        act_name: []
        for act_name in env.env.scene.articulations["robot"].actuators.keys()
    }
    applied_efforts_act = {
        act_name: []
        for act_name in env.env.scene.articulations["robot"].actuators.keys()
    }
    rew_arr = []
    # simulate environment
    m = 0
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():

            ee_quat_root = env.scene.sensors["ee_frame"].data.target_quat_source.squeeze()
            ee_pos_root = env.scene.sensors["ee_frame"].data.target_pos_source.squeeze()
            
            
            ee_pos = transform_points(dict_obs_term_unpack["ee_frame"](env)[:, 0:3], ee_pos_root, ee_quat_root)[0] # Unpack the ee_frame observation
            # Closed Kinematics
            # obj_pos = obs["policy"][0,56:59]  # + torch.Tensor([0.0, 0.0, 0.11]) #4: + torch.Tensor([-0.05, 0.0, 0.1])  #3: + torch.Tensor([-0.05, 0.0, 0.1]) # 2: + torch.Tensor([-0.05, 0.0, 0.1]) #1: + torch.Tensor([-0.1, 0.0, 0.12])
            # target_pos = obs["policy"][0,63:66]
            obj_pos = transform_points(dict_obs_term_unpack["object_position"](env), ee_pos_root, ee_quat_root)[0] + torch.Tensor([0.05, 0.0, 0.0])
            target_pos = transform_points(dict_obs_term_unpack["target_object_position"](env), ee_pos_root, ee_quat_root)[0]
            time_arr.append(env.env.sim.current_time)
            for act_name in applied_efforts_act.keys():
                applied_efforts_act[act_name].append(
                    env.env.scene.articulations["robot"]
                    .actuators[act_name]
                    .applied_effort.tolist()[0]
                )
                computed_efforts_act[act_name].append(
                    env.env.scene.articulations["robot"]
                    .actuators[act_name]
                    .computed_effort.tolist()[0]
                )
            if time_step < 75:
                delta_ee_pos = (obj_pos - ee_pos) * 10
            else:
                delta_ee_pos = (target_pos - ee_pos) * 5
            delta_ee_ang = torch.zeros(3, device=env.unwrapped.device)

            if time_step == 0:
                init_position.append(ee_pos.numpy())

            if time_step < 50:
                # For Closed Kinematics
                # gripper_joint = -torch.ones(7, device=env.unwrapped.device)
                # gripper_joint[2] = 0
                gripper_joint = -torch.ones(10, device=env.unwrapped.device)
                gripper_joint[2] = 0
                ramp = 1
            else:
                if time_step > 50:
                    ramp = 1  # 0.01 * (150 - time_step) + 1
                else:
                    ramp = 1  # 0.5 + np.random.normal(0, 0.8) # 0.01 * (time_step - 45) + np.random.normal(0, 0.8)
                gripper_joint = -torch.ones(10, device=env.unwrapped.device)
                gripper_joint[2] = 0
                # gripper_joint[3:5] = torch.ones(2, device=env.unwrapped.device) * ramp
                # gripper_joint[6] = 1 * ramp
                gripper_joint[3:] = (
                    torch.ones(gripper_joint[3:].shape[0], device=env.unwrapped.device)
                    * ramp
                    * 1.0
                )

            if obj_pos.numpy()[2] > 0.16:
                # save the grasping reference
                # get the body names and joint names
                body_names = env.env.scene.articulations["robot"].body_names
                joint_names = env.env.scene.articulations["robot"].joint_names
                bodies_pos_w = env.env.scene.articulations[
                    "robot"
                ].data.body_pos_w.squeeze()
                bodies_quat_w = env.env.scene.articulations[
                    "robot"
                ].data.body_quat_w.squeeze()
                joint_pos = env.env.scene.articulations[
                    "robot"
                ].data.joint_pos.squeeze()
                obj_pos_w = env.env.scene.rigid_object_collections[
                    "object"
                ].data.object_pos_w[0, env.env.rigid_objects_in_focus[0][0]]
                obj_quat_w = env.env.scene.rigid_object_collections[
                    "object"
                ].data.object_quat_w[0, env.env.rigid_objects_in_focus[0][0]]
                w_pos_obj, w_quat_obj = subtract_frame_transforms(
                    obj_pos_w, obj_quat_w
                )
                # bodies_pos_obj = transform_points(
                #     bodies_pos_w,
                #     w_pos_obj, w_quat_obj
                # )
                bodies_pos_obj, bodies_quat_obj = subtract_frame_transforms(
                    obj_pos_w.unsqueeze(0).repeat((bodies_pos_w.shape[0], 1)),
                    obj_quat_w.unsqueeze(0).repeat((bodies_quat_w.shape[0], 1)),
                    bodies_pos_w,
                    bodies_quat_w,
                )
                d_body_name_pos = {}
                d_body_name_quat = {}
                
                for body_name, b_pos_o, b_quat_o in zip(
                    body_names, bodies_pos_obj.numpy(), bodies_quat_obj.numpy()
                ):
                    d_body_name_pos[body_name] = b_pos_o  # [b_pos_o, b_quat_w]
                    d_body_name_quat[body_name] = b_quat_o
                d_joint_name_pos = {
                    j_name: j_pos
                    for j_name, j_pos in zip(joint_names, joint_pos.numpy())
                }
                
                d_contact_sensor_name_pos = {}
                l_contact_sensor_keys = [key for key in env.env.scene.sensors.keys() if key.find("contact_forces") > -1]
                
                for cs_key in l_contact_sensor_keys:
                    c_contact_sensor = env.env.scene.sensors[cs_key]
                    contact_pos_w = c_contact_sensor.data.contact_pos_w
                    
                    if contact_pos_w is not None and not np.isnan(contact_pos_w.sum().numpy()):
                        contact_pos_obj = transform_points(contact_pos_w.squeeze(1,2), w_pos_obj, w_quat_obj)
                        d_contact_sensor_name_pos[cs_key] = contact_pos_obj.numpy()
                

                grasping_reference = {
                    "qpos": d_joint_name_pos,
                    "key_points": d_body_name_pos,
                    "body_quat_obj": d_body_name_quat,
                    "object_frame": [obj_pos_w.numpy(), obj_quat_w.numpy()],
                    "contact_points_obj": d_contact_sensor_name_pos
                }
                # with open('grasping_reference.npy', 'wb') as f:
                #     np.save(f, np.array([grasping_reference]))
            reward_log.step_update() # Update the reward log
            actions = torch.cat([delta_ee_pos, delta_ee_ang, gripper_joint]).unsqueeze(
                0
            )
            # actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # apply actions

            obs, rew, terminated, truncated, info = env.step(actions)
            time_step += 1
            rew_arr.append(rew)
            # print(time_step, np.round(env.env.sim.current_time, 2), np.round(ramp,3))
            if truncated or terminated:
                reward_log.episode_update() # Update the reward log
                if obj_pos.numpy()[2] > 0.1:
                    succes.append(1)
                else:
                    succes.append(0)
                time_step = 0
                m +=1
                if m > 10:
                    break

    # close the simulator
    env.close() # Close the environment

    reward_log.plot()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
