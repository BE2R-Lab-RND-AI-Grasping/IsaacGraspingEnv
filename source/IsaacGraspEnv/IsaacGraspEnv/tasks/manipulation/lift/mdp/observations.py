# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import copy

import torch
from typing import TYPE_CHECKING, List, Type

from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg, ObservationTermCfg, ManagerTermBase
from isaaclab.utils.math import (
    subtract_frame_transforms,
    transform_points,
    unproject_depth,
)
from isaaclab.sensors import Camera, RayCasterCamera, TiledCamera, FrameTransformer

import open3d as o3d
import numpy as np
from torch import nn
from pathlib import Path


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def show_torch_image(image):
    import matplotlib.pyplot as plt

    plt.imshow(image.permute(0, 1, 2))


def viz_pc_o3d(camera: TiledCamera | Camera | RayCasterCamera):

    color = camera.data.output["rgb"].numpy()
    depth = camera.data.output["distance_to_camera"].numpy()

    width = depth[0].shape[0]
    height = depth[0].shape[1]

    color = o3d.geometry.Image(color[0])
    depth = o3d.geometry.Image(depth[0])

    pinhole_cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx=camera.data.intrinsic_matrices[0, 0, 0],
        fy=camera.data.intrinsic_matrices[0, 1, 1],
        cx=camera.data.intrinsic_matrices[0, 0, 2],
        cy=camera.data.intrinsic_matrices[0, 1, 2],
    )

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_cam_intrinsic)

    # flip the orientation, so it looks upright, not upside-down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    o3d.visualization.draw_geometries([pcd])


def segm_class_label2id(
    camera: TiledCamera | Camera | RayCasterCamera,
):
    idToLabels = camera.data.info[0]["semantic_segmentation"]["idToLabels"]
    label2id = {val["class"]: int(key) for key, val in idToLabels.items()}

    return label2id


# def depth2point_cloud(
#     camera: TiledCamera | Camera | RayCasterCamera,
#     depth_image: torch.Tensor
# ) -> torch.Tensor:
#     height, width = depth_image.shape
#     y, x = torch.meshgrid(torch.arange(height), torch.arange(width))

#     # Convert world unit focal length (mm) to pixel unit. Fx (mm) * w (pixel) / horizontal size sensor (mm). https://ksimek.github.io/2013/08/13/intrinsic/
#     f_x = camera.data.intrinsic_matrices[0,0,0]#camera.cfg.spawn.focal_length * width / camera.cfg.spawn.horizontal_aperture
#     f_y = camera.data.intrinsic_matrices[0,1,1]#camera.cfg.spawn.focal_length * height / camera.cfg.spawn.vertical_aperture


#     c_x = camera.data.intrinsic_matrices[0,0,2]#width / 2
#     c_y = camera.data.intrinsic_matrices[0,1,2]#height / 2


#     Z = depth_image
#     X = (x - c_x) * Z / f_x
#     Y = (y - c_y) * Z / f_y

#     points = torch.stack([X,Y,Z], dim=-1)
#     point_cloud = points.reshape(-1, 3)
#     valid_points = point_cloud[point_cloud[:, 2] > 0]

#     trans_matr = torch.Tensor([[1,0,0],[0,-1,0],[0,0,-1]]).to(valid_points.device)
#     result = torch.matmul(valid_points, trans_matr)

#     # pcd = o3d.geometry.PointCloud()
#     # pcd.points = o3d.utility.Vector3dVector(valid_points.numpy())
#     # pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
#     # o3d.visualization.draw_geometries([pcd])
#     # o3d.io.write_point_cloud(f"./test_point_cloud/frame_1")

#     return result


def depths2pcs(
    camera: TiledCamera | Camera | RayCasterCamera, depth_images: torch.Tensor
) -> torch.Tensor:
    depth_images = depth_images.squeeze()
    n_imgs, height, width = depth_images.shape
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width))

    y = y.repeat(n_imgs, 1).reshape(n_imgs, height, width).to(depth_images.device)
    x = x.repeat(n_imgs, 1).reshape(n_imgs, height, width).to(depth_images.device)

    # Convert world unit focal length (mm) to pixel unit. Fx (mm) * w (pixel) / horizontal size sensor (mm). https://ksimek.github.io/2013/08/13/intrinsic/
    f_x = camera.data.intrinsic_matrices[
        0, 0, 0
    ]  # camera.cfg.spawn.focal_length * width / camera.cfg.spawn.horizontal_aperture
    f_y = camera.data.intrinsic_matrices[
        0, 1, 1
    ]  # camera.cfg.spawn.focal_length * height / camera.cfg.spawn.vertical_aperture

    c_x = camera.data.intrinsic_matrices[0, 0, 2]  # width / 2
    c_y = camera.data.intrinsic_matrices[0, 1, 2]  # height / 2

    X = depth_images
    Y = (x - c_x) * X / f_x
    Z = (y - c_y) * X / f_y

    points = torch.stack([X, Y, Z], dim=-1)
    point_cloud = points.reshape(n_imgs, -1, 3)
    # valid_points = point_cloud[point_cloud[:, :, 2] > 0]

    trans_matr = torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).to(
        point_cloud.device
    )
    result = torch.matmul(point_cloud, trans_matr)
    # result = point_cloud

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(valid_points.numpy())
    # pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    # o3d.visualization.draw_geometries([pcd])
    # o3d.io.write_point_cloud(f"./test_point_cloud/frame_1")

    return result  # .reshape(n_imgs, -1)


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
    ee_pos_b = ee_frame.data.target_pos_source[:, 0]
    ee_quat_b = ee_frame.data.target_quat_source[:, 0]

    ee_frame_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)
    return ee_frame_b


def pos_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    frame_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    frame: RigidObject = env.scene[frame_cfg.name]
    ee_pos_b = frame.data.target_pos_source[:, 0]
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
    object_fingertips_distance = torch.cat(
        [thumb_pos_w, right_pos_w, left_pos_w], dim=1
    )

    return object_fingertips_distance


def depth2point_cloud(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "distance_to_camera",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
    class_filter: list[str] = [],
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
    images[images > 2.5] = 0

    if class_filter:
        segmetation_data = sensor.data.output["semantic_segmentation"]
        label2id = segm_class_label2id(sensor)
        label_masks = torch.zeros_like(segmetation_data)
        for label in class_filter:
            label_masks.add_(torch.eq(segmetation_data, label2id[label]))
        images = images * label_masks

    # depth image conversion
    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(
            images, sensor.data.intrinsic_matrices
        )

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


# def instance_randomize_obj_positions_in_robot_world_frame(
#     env: ManagerBasedRLEnv,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# ) -> torch.Tensor:
#     """The position of the object in the world frame."""
#     if not hasattr(env, "rigid_objects_in_focus"):
#         return torch.full((env.num_envs, 3), fill_value=-1)

#     obj: RigidObjectCollection = env.scene[object_cfg.name]

#     obj_pos_w = []
#     for env_id in range(env.num_envs):
#         obj_pos_w.append(
#             obj.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3]
#         )
#     obj_pos_w = torch.stack(obj_pos_w)

#     return obj_pos_w



def instance_randomize_obj_positions_in_robot_world_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 3), fill_value=-1)

    obj: RigidObjectCollection = env.scene[object_cfg.name]

    obj_pos_w = []
    
    target_pos_b = torch.Tensor([-0.05, 0.0, 0.0]).to(device=env.sim.device)
    
    
    for env_id in range(env.num_envs):
        target_pos_w = transform_points(target_pos_b.unsqueeze(0), obj.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3].unsqueeze(0), obj.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4].unsqueeze(0))
        obj_pos_w.append(
            target_pos_w.squeeze()
            #obj.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3] + target_pos_w.squeeze()
        )
    obj_pos_w = torch.stack(obj_pos_w)

    return obj_pos_w


def instance_randomize_obj_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The orientation of the cubes in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 4), fill_value=-1)

    obj: RigidObjectCollection = env.scene[object_cfg.name]

    obj_quat_w = []
    for env_id in range(env.num_envs):
        obj_quat_w.append(
            obj.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0]]
        )
    obj_quat_w = torch.stack(obj_quat_w)

    return obj_quat_w


class full_obj_point_cloud(ManagerTermBase):

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):

        super().__init__(cfg, env)

        self.list_point_clouds = []
        self.paths_to_pc: list[Path] = [
            Path(path_to_pc) for path_to_pc in cfg.params["path_to_point_clouds"]
        ]

        self.scale_coeff = cfg.params["scale"]
        self.object_cfg = cfg.params["object_cfg"]
        self.object: RigidObjectCollection = env.scene[self.object_cfg.name]

        self.robot_cfg = cfg.params["robot_cfg"]
        self.robot: RigidObject = env.scene[self.robot_cfg.name]

        self.num_pc: int = cfg.params["num_pc"]
        self._load_point_cloud()

        self.last_object_in_focus = copy.deepcopy(env.rigid_objects_in_focus)

        # self.current_obs_pc_in_focus = []
        # self._create_current_point_clound_obs()

    def _create_current_point_clound_obs(self):
        self.current_obs_pc_in_focus = []
        for env_id in range(self.num_envs):
            self.current_obs_pc_in_focus.append(
                self.list_point_clouds[self.last_object_in_focus[env_id][0]].clone()
            )
            
    def _update_current_point_clound_obs(self, obj_in_focus):
        for env_id in range(self.num_envs):
            if self.last_object_in_focus[env_id][0] != obj_in_focus[env_id][0]:
                last_pc = self.current_obs_pc_in_focus[env_id]
                self.current_obs_pc_in_focus[env_id] = self.list_point_clouds[obj_in_focus[env_id][0]].clone()
                del last_pc

    def _load_point_cloud(self):
        for path_to_pc in self.paths_to_pc:
            pcd = o3d.io.read_point_cloud(path_to_pc)
            pcd_reduced = pcd.farthest_point_down_sample(self.num_pc).scale(
                self.scale_coeff, np.zeros(3)
            )
            point_cloud = torch.from_numpy(np.array(pcd_reduced.points)).float()
            self.list_point_clouds.append(point_cloud.to(self.device))

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        path_to_point_clouds: str,
        object_cfg: SceneEntityCfg,
        robot_cfg: SceneEntityCfg,
        scale: float,
        num_pc: int,
    ):
        """The position of the object in the world frame."""
        if not hasattr(env, "rigid_objects_in_focus"):
            return torch.full((env.num_envs, self.num_pc), fill_value=-1)

        # self._update_current_point_clound_obs(env.rigid_objects_in_focus)
        
        obj_p_w_o = instance_randomize_obj_positions_in_robot_world_frame(
            env, self.object_cfg
        )
        obj_quat_w_o = instance_randomize_obj_orientations_in_world_frame(
            env, self.object_cfg
        )
        obj_pos_b, obj_quat_b = subtract_frame_transforms(
            self.robot.data.root_state_w[:, :3],
            self.robot.data.root_state_w[:, 3:7],
            obj_p_w_o,
            obj_quat_w_o,
        )
        points_b = []
        for env_id in range(env.num_envs):

            # if self.last_object_in_focus[env_id] == env.rigid_objects_in_focus[env_id]:
            points_b.append(transform_points(  # observed points in world frame
                self.list_point_clouds[
                    env.rigid_objects_in_focus[env_id][0]
                ],
                obj_pos_b[env_id],
                obj_quat_b[env_id],
            ))
        points_b = torch.stack(points_b)
        return points_b

def instance_randomize_obj_positions_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 3), fill_value=-1)

    robot: RigidObject = env.scene[robot_cfg.name]

    obj: RigidObjectCollection = env.scene[object_cfg.name]
    target_pos_obj_b = torch.Tensor([-0.05, 0.0, 0.0]).to(env.sim.device)
    
    obj_pos_w = []
    for env_id in range(env.num_envs):
        target_pos_w = transform_points(target_pos_obj_b.unsqueeze(0), obj.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3].unsqueeze(0), obj.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4].unsqueeze(0))
        obj_pos_w.append(
            target_pos_w.squeeze()
            #obj.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0]] + target_pos_w.squeeze()
        )
        
    obj_pos_w = torch.stack(obj_pos_w)

    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], obj_pos_w
    )
    
    
    
    object_pos_b = object_pos_b 
    return object_pos_b


def instance_randomize_obj_orientations_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The orientation of the cubes in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObjectCollection = env.scene[object_cfg.name]

    obj_quat_w = []
    obj_pos_w = []
    for env_id in range(env.num_envs):
        obj_quat_w.append(
            obj.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4]
        )
        obj_pos_w.append(
            obj.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3]
        )
    obj_quat_w = torch.stack(obj_quat_w)
    obj_pos_w = torch.stack(obj_pos_w)

    _, object_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        obj_pos_w,
        obj_quat_w,
    )
    return object_quat_b
