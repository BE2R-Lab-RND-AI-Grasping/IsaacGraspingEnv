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
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Iiwa-IK-Rel-v0", help="Name of the task.")
# append AppLauncher cli args

parser.add_argument(
    "--dataset_path",
    type=str,
    default="/home/yefim-home/Documents/work/IsaacGraspingEnv/source/IsaacGraspEnv/IsaacGraspEnv/assets/data/HANDEL/screwdrivers", #"wrenches",#None,
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

def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device="cpu", num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        # args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # env_cfg.terminations.time_out = None
    env_cfg.episode_length_s = 3
    if args_cli.model_filter:
        dt_models_filter = args_cli.model_filter.replace(" ", "").split(",")
    else:
        dt_models_filter = args_cli.model_filter
        

    env_cfg.scene.object.rigid_objects =  load_object_dataset(
        args_cli.dataset_path,
        args_cli.usd_file_name,
        dt_models_filter
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
                prim_path="{ENV_REGEX_NS}/Object_model_4",
                name="target_model_4",
                offset=OffsetCfg(
                    pos=[-0.05, 0.0, 0.0],
                ),
            ),
        ],
    )

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.env.sim.set_camera_view([2.5,1, 1], [0.0, 0.0,0.0])
    env.env.sim.set_render_mode(env.env.sim.RenderMode.FULL_RENDERING)
    
    env.scene.articulations["robot"].root_physx_view.prim_paths
    
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
    
    init_position = []
    succes = []
    effort_limits = {act_name:env.env.scene.articulations["robot"].actuators[act_name].effort_limit.tolist()[0]
                    for act_name in  env.env.scene.articulations["robot"].actuators.keys()}
    computed_efforts_act = {act_name:[] for act_name in  env.env.scene.articulations["robot"].actuators.keys()}
    applied_efforts_act = {act_name:[] for act_name in  env.env.scene.articulations["robot"].actuators.keys()}
    rew_arr = []
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            ee_pos = obs["policy"][0,0:3]
            obj_pos = obs["policy"][0,56:59]+ torch.Tensor([0.0, 0.0, 0.11]) #4: + torch.Tensor([-0.05, 0.0, 0.1])  #3: + torch.Tensor([-0.05, 0.0, 0.1]) # 2: + torch.Tensor([-0.05, 0.0, 0.1]) #1: + torch.Tensor([-0.1, 0.0, 0.12])
            target_pos = obs["policy"][0,63:66]
            time_arr.append(env.env.sim.current_time)
            for act_name in applied_efforts_act.keys():
                applied_efforts_act[act_name].append(env.env.scene.articulations["robot"].actuators[act_name].applied_effort.tolist()[0])
                computed_efforts_act[act_name].append(env.env.scene.articulations["robot"].actuators[act_name].computed_effort.tolist()[0])
            if time_step < 100:
                delta_ee_pos = (obj_pos - ee_pos) / 2
            else:
                delta_ee_pos = (target_pos - ee_pos) / 2
            delta_ee_ang = torch.zeros(3, device=env.unwrapped.device)

            if time_step == 0:
                init_position.append(ee_pos.numpy())

            if time_step < 45:
                gripper_joint = -torch.ones(7, device=env.unwrapped.device)
                gripper_joint[2] = 0
                ramp = 1
            else:
                if time_step > 150:
                    ramp = 1#0.01 * (150 - time_step) + 1
                else:
                    ramp = 1# 0.5 + np.random.normal(0, 0.8) # 0.01 * (time_step - 45) + np.random.normal(0, 0.8)
                gripper_joint = -torch.ones(7, device=env.unwrapped.device)
                gripper_joint[2] = 0
                gripper_joint[3:5] = torch.ones(2, device=env.unwrapped.device) * ramp
                gripper_joint[6] = 1 * ramp

            
            actions = torch.cat([delta_ee_pos, delta_ee_ang, gripper_joint]).unsqueeze(0)
            # actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # apply actions
            obs, rew, terminated, truncated, info =  env.step(actions)
            time_step +=1
            rew_arr.append(rew)
            # print(time_step, np.round(env.env.sim.current_time, 2), np.round(ramp,3))
            if truncated or terminated:
                if obj_pos.numpy()[2] > 0.2:
                    succes.append(1)
                else:
                    succes.append(0)
                time_step =0
                # break

    # close the simulator
    env.close()
    
    for act_name in applied_efforts_act.keys():
        plt.title(act_name)
        for i in range(len(applied_efforts_act[act_name][0])):
            plt.plot(time_arr, np.array(computed_efforts_act[act_name])[:,i], "--", linewidth=1.5, label=f"computed_{i}")
            plt.plot(time_arr, np.array(applied_efforts_act[act_name])[:,i], linewidth=1.5, label=f"applied_{i}")
            plt.plot([time_arr[0], time_arr[-1]], [effort_limits[act_name][i] for __ in range(2)], linewidth=1.5, label=f"max_limit{i}")
            plt.plot([time_arr[0], time_arr[-1]], [-effort_limits[act_name][i] for __ in range(2)], linewidth=1.5, label=f"min_limit{i}")
        plt.xlabel("time, s")
        plt.ylabel("Effort, Nm")
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
