# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, transform_points, unproject_depth
from isaaclab.sensors import Camera, RayCasterCamera, TiledCamera

import open3d as o3d
import numpy as np

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def show_torch_image(image):
    import matplotlib.pyplot as plt
    plt.imshow(image.permute(0, 1, 2))


def viz_pc_o3d(camera: TiledCamera | Camera | RayCasterCamera):
    
    
    color = camera.data.output["rgb"].numpy()
    depth = camera.data.output['distance_to_camera'].numpy()
    
    width = depth[0].shape[0]
    height = depth[0].shape[1]
    
    color = o3d.geometry.Image(color[0])
    depth = o3d.geometry.Image(depth[0])
    
    pinhole_cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width= width,
        height= height,
        fx=camera.data.intrinsic_matrices[0,0,0], fy=camera.data.intrinsic_matrices[0,1,1],
        cx=camera.data.intrinsic_matrices[0,0,2], cy=camera.data.intrinsic_matrices[0,1,2]
    )
    
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity = False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_cam_intrinsic)

    # flip the orientation, so it looks upright, not upside-down
    pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

    o3d.visualization.draw_geometries([pcd])

def segm_class_label2id(
    camera: TiledCamera | Camera | RayCasterCamera,
    ):
    idToLabels = camera.data.info[0]["semantic_segmentation"]["idToLabels"]
    label2id = {val["class"]: int(key) for key, val in idToLabels.items()}
    
    return label2id

def depth2point_cloud(
    camera: TiledCamera | Camera | RayCasterCamera,
    depth_image: torch.Tensor
) -> torch.Tensor:
    height, width = depth_image.shape
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width))
    
    # Convert world unit focal length (mm) to pixel unit. Fx (mm) * w (pixel) / horizontal size sensor (mm). https://ksimek.github.io/2013/08/13/intrinsic/
    f_x = camera.data.intrinsic_matrices[0,0,0]#camera.cfg.spawn.focal_length * width / camera.cfg.spawn.horizontal_aperture
    f_y = camera.data.intrinsic_matrices[0,1,1]#camera.cfg.spawn.focal_length * height / camera.cfg.spawn.vertical_aperture
    
    
    c_x = camera.data.intrinsic_matrices[0,0,2]#width / 2
    c_y = camera.data.intrinsic_matrices[0,1,2]#height / 2
    
    
    Z = depth_image
    X = (x - c_x) * Z / f_x
    Y = (y - c_y) * Z / f_y
    
    points = torch.stack([X,Y,Z], dim=-1)
    point_cloud = points.reshape(-1, 3)
    valid_points = point_cloud[point_cloud[:, 2] > 0]
    
    trans_matr = torch.Tensor([[1,0,0],[0,-1,0],[0,0,-1]]).to(valid_points.device)
    result = torch.matmul(valid_points, trans_matr)
    
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(valid_points.numpy())
    # pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    # o3d.visualization.draw_geometries([pcd])
    # o3d.io.write_point_cloud(f"./test_point_cloud/frame_1")
    
    return result
    
def depths2pcs(
    camera: TiledCamera | Camera | RayCasterCamera,
    depth_images: torch.Tensor
) -> torch.Tensor:
    depth_images = depth_images.squeeze()
    n_imgs, height, width = depth_images.shape
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width))
    
    y = y.repeat(n_imgs, 1).reshape(n_imgs, height, width)
    x = x.repeat(n_imgs, 1).reshape(n_imgs, height, width)
    
    # Convert world unit focal length (mm) to pixel unit. Fx (mm) * w (pixel) / horizontal size sensor (mm). https://ksimek.github.io/2013/08/13/intrinsic/
    f_x = camera.data.intrinsic_matrices[0,0,0]#camera.cfg.spawn.focal_length * width / camera.cfg.spawn.horizontal_aperture
    f_y = camera.data.intrinsic_matrices[0,1,1]#camera.cfg.spawn.focal_length * height / camera.cfg.spawn.vertical_aperture
    
    
    c_x = camera.data.intrinsic_matrices[0,0,2]#width / 2
    c_y = camera.data.intrinsic_matrices[0,1,2]#height / 2
    
    
    Z = depth_images
    X = (x - c_x) * Z / f_x
    Y = (y - c_y) * Z / f_y
    
    points = torch.stack([X,Y,Z], dim=-1)
    point_cloud = points.reshape(n_imgs,-1, 3)
    # valid_points = point_cloud[point_cloud[:, :, 2] > 0]

    trans_matr = torch.Tensor([[1,0,0],[0,-1,0],[0,0,-1]]).to(point_cloud.device)
    result = torch.matmul(point_cloud, trans_matr)
    
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(valid_points.numpy())
    # pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    # o3d.visualization.draw_geometries([pcd])
    # o3d.io.write_point_cloud(f"./test_point_cloud/frame_1")
    
    return result

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
    class_filter: list[str] = []
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
    # images[images > 3] = 0
    
    if class_filter:
        segmetation_data = sensor.data.output["semantic_segmentation"]
        label2id = segm_class_label2id(sensor)
        label_masks = torch.zeros_like(segmetation_data)  
        for label in class_filter:
            label_masks.add_(torch.eq(segmetation_data, label2id[label]))
        images = images * label_masks
    
    
    # depth image conversion
    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

    pcs = depths2pcs(sensor, images)
    
    
    # rgb/depth image normalization
    if normalize:
        if data_type == "rgb":
            images = images.float() / 255.0
            mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
            images -= mean_tensor
        elif "distance_to" in data_type or "depth" in data_type:
            images[images == float("inf")] = 0

    return pcs.clone()