# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b

def object_quat_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_quat_w = object.data.root_quat_w[:, :4]

    return object_quat_w

def frame_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    ee_frame: RigidObject = env.scene[ee_frame_cfg.name]
    ee_pos_b = ee_frame.data.target_pos_source[:,0]
    ee_quat_b = ee_frame.data.target_quat_source[:,0]
    
    ee_frame_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)
    return ee_frame_b

def pos_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    frame_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    frame: RigidObject = env.scene[frame_cfg.name]
    ee_pos_b = frame.data.target_pos_source[:,0]
    return ee_pos_b


def pos_fingertips_root_frame(
    env: ManagerBasedRLEnv,
    thumb_ft_frame_cfg: SceneEntityCfg = SceneEntityCfg("thumb_ft_frame"),
    right_ft_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ft_frame"),
    left_ft_frame_cfg: SceneEntityCfg = SceneEntityCfg("left_ft_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    thumb_ft_frame: FrameTransformer = env.scene[thumb_ft_frame_cfg.name]
    right_ft_frame: FrameTransformer = env.scene[right_ft_frame_cfg.name]
    left_ft_frame: FrameTransformer = env.scene[left_ft_frame_cfg.name]
    # End-effector position: (num_envs, 3)
    thumb_pos_w = thumb_ft_frame.data.target_pos_source[..., 0, :]
    right_pos_w = right_ft_frame.data.target_pos_source[..., 0, :]
    left_pos_w = left_ft_frame.data.target_pos_source[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_fingertips_distance = torch.cat([
        thumb_pos_w,
        right_pos_w,
        left_pos_w

    ], dim=1 )
    
    return object_fingertips_distance
