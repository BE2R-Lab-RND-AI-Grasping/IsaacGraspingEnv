# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors import Camera, RayCasterCamera, TiledCamera

import open3d as o3d
import numpy as np

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def depth2point_cloud(
    camera: TiledCamera | Camera | RayCasterCamera,
    depth_image: torch.Tensor
) -> torch.Tensor:
    height, width = depth_image.shape
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width))
    
    f_x = camera.cfg.spawn.focal_length
    f_y = camera.cfg.spawn.focal_length
    
    
    c_x = width / 2
    c_y = height / 2
    
    
    Z = depth_image
    X = (x - c_x) * Z / f_x
    Y = (y - c_y) * Z / f_y
    
    points = torch.stack([X,Y,Z], dim=-1)
    point_cloud = points.reshape(-1, 3)
    valid_points = point_cloud[point_cloud[:, 2] > 0]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points.numpy())
    o3d.visualization.draw_geometries([pcd])
    
    return valid_points
    


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


def point_cloud(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        normalize: Whether to normalize the images. This depends on the selected data type.
            Defaults to True.

    Returns:
        The images produced at the last time-step
    """
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # obtain the input image
    images = sensor.data.output[data_type]
    images[images == float("inf")] = 0
    images[images > 3] = 0
    depth2point_cloud(sensor, images[0].squeeze())
    
    # depth image conversion
    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

    # rgb/depth image normalization
    if normalize:
        if data_type == "rgb":
            images = images.float() / 255.0
            mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
            images -= mean_tensor
        elif "distance_to" in data_type or "depth" in data_type:
            images[images == float("inf")] = 0

    return images.clone()