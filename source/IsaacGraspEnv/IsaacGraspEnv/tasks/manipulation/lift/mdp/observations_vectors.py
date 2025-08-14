# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import copy

from source.IsaacGraspEnv.IsaacGraspEnv.debug_function.viz_frames import (
    o3d_viz_body_key_points_obj_frames,
)
import torch
from typing import TYPE_CHECKING, List, Type

from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg, ObservationTermCfg, ManagerTermBase
from isaaclab.utils.math import (
    subtract_frame_transforms,
    transform_points,
    skew_symmetric_matrix,
    quat_mul,
)
from isaaclab.sensors import ContactSensor
from kornia.geometry.liegroup import Se3
from kornia.geometry.quaternion import Quaternion
import open3d as o3d
import numpy as np
from torch import nn
from pathlib import Path

from .observations import full_obj_point_cloud

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv




def instance_randomize_obj_positions_in_robot_ee_frame(
    env: ManagerBasedRLEnv,
    frame_cfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The orientation of the cubes in the robot end-effector frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 3), fill_value=-1)

    ee_frame: RigidObject = env.scene[frame_cfg.name]
    obj: RigidObjectCollection = env.scene[object_cfg.name]

    object_pos_w = obj.data.object_pos_w[
        list(range(env.num_envs)), np.squeeze(env.rigid_objects_in_focus)
    ]

    object_quat_w = obj.data.object_quat_w[
        list(range(env.num_envs)), np.squeeze(env.rigid_objects_in_focus)
    ]

    object_pos_ee, __ = subtract_frame_transforms(
        ee_frame.data.target_pos_w.squeeze(1),
        ee_frame.data.target_quat_w.squeeze(1),
        object_pos_w,
        object_quat_w,
    )
    return object_pos_ee


def instance_randomize_obj_orientations_in_robot_ee_frame(
    env: ManagerBasedRLEnv,
    frame_cfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The orientation of the cubes in the robot end-effector frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 4), fill_value=-1)

    ee_frame: RigidObject = env.scene[frame_cfg.name]
    obj: RigidObjectCollection = env.scene[object_cfg.name]

    object_pos_w = obj.data.object_pos_w[
        list(range(env.num_envs)), np.squeeze(env.rigid_objects_in_focus)
    ]

    object_quat_w = obj.data.object_quat_w[
        list(range(env.num_envs)), np.squeeze(env.rigid_objects_in_focus)
    ]

    __, object_quat_ee = subtract_frame_transforms(
        ee_frame.data.target_pos_w.squeeze(1),
        ee_frame.data.target_quat_w.squeeze(1),
        object_pos_w,
        object_quat_w,
    )
    return object_quat_ee


def instance_randomize_obj_vel_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg:  SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Calculate the velocity of the objects in the robot frame.
        Calculating is based on twist transformation from the world frame to the robot frame.
        Simple working example of the transformation in file `scripts/testing/exp_2_general_vel.py`
    Args:
        env (ManagerBasedRLEnv): The environment instance.
        robot_cfg (SceneEntityCfg): The configuration for the robot.
        object_cfg (SceneEntityCfg, optional): The configuration for the object. Defaults to SceneEntityCfg("object").
    """
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 6), fill_value=-1)

    obj: RigidObjectCollection = env.scene[object_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]


    body_velocity_object = []
    
    for env_id in range(env.num_envs):
        obj_pos_w = obj.data.object_link_pos_w[env_id, env.rigid_objects_in_focus[env_id][0]]
        obj_quat_w = obj.data.object_link_quat_w[env_id, env.rigid_objects_in_focus[env_id][0]]
        kornia_obj_quat_w = Quaternion(obj_quat_w)
        H_w_obj = Se3(kornia_obj_quat_w, obj_pos_w)
        
        obj_ang_vel_w = (obj.data.object_ang_vel_w[env_id, env.rigid_objects_in_focus[env_id][0]]
                                / 180
                                * np.pi)
        obj_lin_vel_w = obj.data.object_lin_vel_w[env_id, env.rigid_objects_in_focus[env_id][0]]
        
        
        obj_sp_vel_w = - skew_symmetric_matrix(obj_ang_vel_w)[0] @ obj_pos_w + obj_lin_vel_w
        
        w_twist_w_obj = torch.cat([obj_sp_vel_w, obj_ang_vel_w])
        
        body_pos_w = robot.data.body_pos_w[env_id, robot_cfg.body_ids].squeeze()
        body_quat_w = robot.data.body_quat_w[env_id, robot_cfg.body_ids].squeeze()
        kornia_body_quat_w = Quaternion(body_quat_w)
        H_w_body = Se3(kornia_body_quat_w, body_pos_w)
        
        body_vel_w = robot.data.body_vel_w[env_id, robot_cfg.body_ids].squeeze()
        body_lin_vel_w = body_vel_w[:3]
        body_ang_vel_w = body_vel_w[3:]
        
        body_sp_vel_w = - skew_symmetric_matrix(body_ang_vel_w)[0] @ body_pos_w + body_lin_vel_w
        w_twist_body_w = torch.cat([body_sp_vel_w, body_ang_vel_w])
        
        inv_Ad_H_w_body = H_w_body.inverse().adjoint()
        
        body_twist_body_w = - inv_Ad_H_w_body @ w_twist_body_w
        
        body_twist_body_obj = body_twist_body_w + inv_Ad_H_w_body @ w_twist_w_obj
        
        body_lin_vel_body_obj = Se3.exp(body_twist_body_obj).matrix() @ H_w_body.inverse().matrix() @ torch.cat([obj_pos_w, torch.ones(1)])
        body_ang_vel_body_obj = body_twist_body_obj[3:]
        
        body_vel_obj = torch.cat([body_lin_vel_body_obj[:3], body_ang_vel_body_obj])
        
        body_velocity_object.append(body_vel_obj)
        
    body_velocity_object = torch.stack(body_velocity_object)

    return body_velocity_object

def robot_body_vel_in_body_frame(
    env: ManagerBasedRLEnv,
    robot_cfg:  SceneEntityCfg,
    ) -> torch.Tensor:
    
    robot: RigidObject = env.scene[robot_cfg.name]
    
    body_velocity_body = []
    
    for env_id in range(env.num_envs):
        body_pos_w = robot.data.body_pos_w[env_id, robot_cfg.body_ids].squeeze()
        body_quat_w = robot.data.body_quat_w[env_id, robot_cfg.body_ids].squeeze()
        kornia_body_quat_w = Quaternion(body_quat_w)
        H_w_body = Se3(kornia_body_quat_w, body_pos_w)
        
        body_vel_w = robot.data.body_vel_w[env_id, robot_cfg.body_ids].squeeze()
        body_lin_vel_w = body_vel_w[:3]
        body_ang_vel_w = body_vel_w[3:]
                
        body_sp_vel_w = - skew_symmetric_matrix(body_ang_vel_w)[0] @ body_pos_w + body_lin_vel_w
        w_twist_body_w = torch.cat([body_sp_vel_w, body_ang_vel_w])
        
        b_twist_body_w = H_w_body.inverse().adjoint() @ w_twist_body_w

        body_vel_w_b = (Se3.exp(b_twist_body_w).matrix() @ torch.cat([torch.zeros(3), torch.ones(1)]))[:3]

        body_velocity_body.append(torch.cat([body_vel_w_b, b_twist_body_w[3:]]))

    return torch.stack(body_velocity_body)


class instance_randomize_obj_displacement(ManagerTermBase):
    
    def __init__(self,
        cfg: ObservationTermCfg,
        env: ManagerBasedRLEnv,
    ):
        super().__init__(cfg, env)
        
        tuple_initial_obj_pos_b = cfg.params["initial_object_position_base"]
        self.initial_objects_position_b = torch.Tensor(tuple_initial_obj_pos_b).to(env.device)
        
        self.initial_objects_position_b.repeat(env.num_envs, 1)
        
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        frame_cfg: SceneEntityCfg,
        initial_object_position_base: tuple,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        ) -> torch.Tensor:
        
        ee_frame: RigidObject = env.scene[frame_cfg.name]
        
        
        object_pos_ee = instance_randomize_obj_positions_in_robot_ee_frame(env, frame_cfg, object_cfg)
        
        init_obj_pos_ee, __ = subtract_frame_transforms(
            ee_frame.data.target_pos_source.squeeze(1),
            ee_frame.data.target_quat_source.squeeze(1),
            self.initial_objects_position_b,
            )
        
        return object_pos_ee - init_obj_pos_ee
    


def vectors_joint_hand_object_frame(
    env: ManagerBasedRLEnv,
    frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Calculate the vectors from the robot hand to the object in the robot end-effector frame."""

    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 3))

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: RigidObject = env.scene[frame_cfg.name]
    object_collection = env.scene[object_cfg.name]

    object_pos_w = object_collection.data.object_pos_w[
        list(range(env.num_envs)), np.squeeze(env.rigid_objects_in_focus)
    ]

    object_pos_ee, __ = subtract_frame_transforms(
        ee_frame.data.target_pos_w.squeeze(),
        ee_frame.data.target_quat_w.squeeze(),
        object_pos_w,
    )
    w_pos_ee, w_quat_ee = subtract_frame_transforms(
        ee_frame.data.target_pos_w.squeeze(), ee_frame.data.target_quat_w.squeeze()
    )

    bodies_pos_ee = transform_points(
        robot.data.body_pos_w[:, robot_cfg.body_ids, :3],
        w_pos_ee.squeeze(),
        w_quat_ee.squeeze(),
    )

    vec_hand_bodies_obj_ee = object_pos_ee.unsqueeze(1) - bodies_pos_ee

    vec_hand_bodies_obj_ee = vec_hand_bodies_obj_ee.flatten(1)

    return vec_hand_bodies_obj_ee / 2


class instance_vectors_joint_hand_object_full_pc(full_obj_point_cloud):

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        """Class observation term to compute the vectors from the robot hand to full object point cloud in the robot end-effector frame."""
        super().__init__(cfg, env)

        self.ee_frame_cfg = cfg.params["frame_cfg"]
        self.ee_frame: RigidObject = env.scene[self.ee_frame_cfg.name]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        frame_cfg: SceneEntityCfg,
        path_to_point_clouds: str,
        scale: float,
        num_pc: int,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Calculate the vectors from the robot hand to the full object point cloud in the robot end-effector frame.

        Args:
            env (ManagerBasedRLEnv): The environment instance.
            frame_cfg (SceneEntityCfg): The configuration for the frame.
            path_to_point_clouds (str): The path to the full point clouds.
            scale (float): The scale factor for the point clouds.
            num_pc (int): The number of points in point clouds.
            object_cfg (SceneEntityCfg, optional): The configuration for the object. Defaults to SceneEntityCfg("object").
            robot_cfg (SceneEntityCfg, optional): The configuration for the robot. Defaults to SceneEntityCfg("robot").
    
        Returns:
            torch.Tensor: The vectors from the robot hand to the full object point cloud in the robot end-effector frame.
        """
        if not hasattr(env, "rigid_objects_in_focus"):
            return torch.full((env.num_envs, 3))

        w_pos_ee, w_quat_ee = subtract_frame_transforms(
            self.ee_frame.data.target_pos_w.squeeze(),
            self.ee_frame.data.target_quat_w.squeeze(),
        )

        bodies_pos_ee = transform_points(
            self.robot.data.body_pos_w[:, robot_cfg.body_ids, :3],
            w_pos_ee.squeeze(),
            w_quat_ee.squeeze(),
        )

        object_pos_ee, object_quat_ee = subtract_frame_transforms(
            self.object.data.object_pos_w,
            self.object.data.object_quat_w,
            self.ee_frame.data.target_pos_w.expand(-1, self.object.num_objects, -1),
            self.ee_frame.data.target_quat_w.expand(-1, self.object.num_objects, -1),
        )

        list_vectors2closest_pc = []

        for env_id in range(env.num_envs):
            pc_pos_ee = transform_points(
                self.list_point_clouds[env.rigid_objects_in_focus[env_id][0]],
                object_pos_ee[env_id, env.rigid_objects_in_focus[env_id][0]].squeeze(),
                object_quat_ee[env_id, env.rigid_objects_in_focus[env_id][0]].squeeze(),
            )

            vectors = pc_pos_ee.unsqueeze(0).expand(
                bodies_pos_ee[env_id].shape[0], -1, -1
            ) - bodies_pos_ee[env_id].unsqueeze(1)
            min_indeces = torch.linalg.norm(vectors, dim=2).argmin(1)
            vectors2closest_pc = vectors[
                list(range(bodies_pos_ee[env_id].shape[0])), min_indeces, :
            ].flatten()

            list_vectors2closest_pc.append(vectors2closest_pc)

        tensor_vectors2closest_pc = torch.stack(list_vectors2closest_pc)

        return tensor_vectors2closest_pc


class instance_vectors_joint_hand_key_points(ManagerTermBase):

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        """Class observation term to compute the vectors from the robot hand to the key points in the robot end-effector frame."""
        super().__init__(cfg, env)

        self.body_frame_key = cfg.params["body_frame_key"]
        self.joint_position_key = cfg.params["joint_position_key"]

        self.object_cfg = cfg.params["object_cfg"]
        self.object: RigidObjectCollection = env.scene[self.object_cfg.name]

        self.robot_cfg = cfg.params["robot_cfg"]
        self.robot: RigidObject = env.scene[self.robot_cfg.name]

        self.last_object_in_focus = copy.deepcopy(env.rigid_objects_in_focus)
        # self.current_obs_pc_in_focus = []
        # self._create_current_point_clound_obs()

        self.ee_frame_cfg = cfg.params["frame_cfg"]
        self.ee_frame: RigidObject = env.scene[self.ee_frame_cfg.name]

        self.obj_key_points_ref = (
            self.mapping_reference_n_sim_hand(cfg.params["grasping_reference_path"])
            .repeat(env.num_envs, 1, 1)
            .to(env.device)
        )

    def mapping_reference_n_sim_hand(self, grasping_reference_path):

        with open(grasping_reference_path, "rb") as f:

            grasping_reference = np.load(f, allow_pickle=True)

        robot_body_names = [self.robot.body_names[m] for m in self.robot_cfg.body_ids]

        ordered_grasping_reference = []
        for grasp_ref in grasping_reference:
            one_obj_ref = []
            for body_name in robot_body_names:
                one_obj_ref.append(grasp_ref[self.body_frame_key][body_name])

            ordered_grasping_reference.append(one_obj_ref)

        return torch.tensor(ordered_grasping_reference)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        frame_cfg: SceneEntityCfg,
        grasping_reference_path: str,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        body_frame_key: str = "key_points",
        joint_position_key: str = "qpos",
    ) -> torch.Tensor:
        """Compute the vectors from the robot hand to the key points in the robot end-effector frame. The key points are defined in the grasping reference file.
        joint_position_key is not used in this function, but it is required for the config to be valid.

        Args:
            env (ManagerBasedRLEnv): The environment instance.
            frame_cfg (SceneEntityCfg): The configuration for the frame.
            grasping_reference_path (str): The path to the grasping reference file. The file should contain list of dictionaries with keys defined in body_frame_key.
            object_cfg (SceneEntityCfg, optional): The configuration for the object. Defaults to SceneEntityCfg("object").
            robot_cfg (SceneEntityCfg, optional): The configuration for the robot. Defaults to SceneEntityCfg("robot").
            body_frame_key (str, optional): The dictionary key for the body frame in the grasping reference file. Defaults to "key_points".
            joint_position_key (str, optional): The dictionary key for the joint position in the grasping reference file. Defaults to "qpos".

        Returns:
            torch.Tensor: The computed vectors from the robot hand to the key points in the robot end-effector frame.
        """

        if not hasattr(env, "rigid_objects_in_focus"):
            return torch.full((env.num_envs, 3 * self.object_cfg.num_bodies))

        w_pos_ee, w_quat_ee = subtract_frame_transforms(
            self.ee_frame.data.target_pos_w.squeeze(),
            self.ee_frame.data.target_quat_w.squeeze(),
        )

        bodies_pos_ee = transform_points(
            self.robot.data.body_pos_w[:, robot_cfg.body_ids, :3],
            w_pos_ee.squeeze(),
            w_quat_ee.squeeze(),
        )

        object_pos_ee, object_quat_ee = subtract_frame_transforms(
            self.ee_frame.data.target_pos_w.expand(-1, self.object.num_objects, -1),
            self.ee_frame.data.target_quat_w.expand(-1, self.object.num_objects, -1),
            self.object.data.object_pos_w,
            self.object.data.object_quat_w,
        )

        list_vectors2grasping_reference = []

        for env_id in range(env.num_envs):
            ee_key_points_ref = transform_points(
                self.obj_key_points_ref[env.rigid_objects_in_focus[env_id][0]],
                object_pos_ee[env_id, env.rigid_objects_in_focus[env_id][0]].squeeze(),
                object_quat_ee[env_id, env.rigid_objects_in_focus[env_id][0]].squeeze(),
            )

            vectors = ee_key_points_ref - bodies_pos_ee[env_id]

            list_vectors2grasping_reference.append(vectors.flatten())

        tensor_vectors2closest_pc = torch.stack(list_vectors2grasping_reference)

        # For Debug
        # o3d_viz_body_key_points_obj_frames(
        #     bodies_pos_ee[env_id],
        #     ee_key_points_ref,
        #     "/home/yefim-home/Downloads/Telegram Desktop/dataset/power_drills/model_1/object_convex_decomposition_meter_unit.obj",
        #     object_pos_ee[env.rigid_objects_in_focus[env_id][0], 0],
        #     object_quat_ee[env.rigid_objects_in_focus[env_id][0], 0]
        # )
        return tensor_vectors2closest_pc


class instance_target_end_effector_orientation(ManagerTermBase):

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        """Class observation term to compute the target end-effector orientation in the robot end-effector frame.
        The target orientation is defined in the grasping reference file. 
        Args:
            cfg (ObservationTermCfg): The configuration for the observation term.
            env (ManagerBasedRLEnv): The environment instance.
        """
        super().__init__(cfg, env)

        self.body_quat_key = cfg.params["body_quat_key"]

        self.object_cfg = cfg.params["object_cfg"]
        self.object: RigidObjectCollection = env.scene[self.object_cfg.name]

        self.robot_cfg = cfg.params["robot_cfg"]
        self.robot: RigidObject = env.scene[self.robot_cfg.name]

        self.ee_frame_cfg = cfg.params["frame_cfg"]
        self.ee_frame: RigidObject = env.scene[self.ee_frame_cfg.name]

        self.ee_quat_reference_obj = (
            self.mapping_reference_n_sim_hand(cfg.params["grasping_reference_path"])
            .repeat(env.num_envs, 1, 1)
            .to(env.device)
        )

    def mapping_reference_n_sim_hand(self, grasping_reference_path):

        with open(grasping_reference_path, "rb") as f:

            grasping_reference = np.load(f, allow_pickle=True)

        robot_body_names = [self.robot.body_names[m] for m in self.robot_cfg.body_ids]

        ordered_grasping_reference = []
        for grasp_ref in grasping_reference:
            one_obj_ref = []
            for body_name in robot_body_names:
                one_obj_ref.append(grasp_ref[self.body_quat_key][body_name])

            ordered_grasping_reference.append(one_obj_ref)

        return torch.Tensor(ordered_grasping_reference)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        frame_cfg: SceneEntityCfg,
        grasping_reference_path: str,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        body_quat_key: str = "body_quat_obj",
    ) -> torch.Tensor:
        """Compute the target end-effector orientation in the robot end-effector frame.
        Grasping reference file should contain list of dictionaries with keys defined in body_quat_key.
        In the file, the key should contain the quaternion of the end-effector in the object frame.
        Args:
            env (ManagerBasedRLEnv): The environment instance.
            frame_cfg (SceneEntityCfg): The configuration for the frame.
            grasping_reference_path (str): The path to the grasping reference file.
            object_cfg (SceneEntityCfg, optional): The configuration for the object. Defaults to SceneEntityCfg("object").
            robot_cfg (SceneEntityCfg, optional): The configuration for the robot. Defaults to SceneEntityCfg("robot").
            body_quat_key (str, optional): The dictionary key for the body quaternion in object frame in the grasping reference file. Defaults to "body_quat_obj".
        Returns:
            torch.Tensor: The target end-effector orientation in the robot end-effector frame.
        """
        if not hasattr(env, "rigid_objects_in_focus"):
            return torch.full((env.num_envs, 4))

        object_pos_ee, object_quat_ee = subtract_frame_transforms(
            self.ee_frame.data.target_pos_w.expand(-1, self.object.num_objects, -1),
            self.ee_frame.data.target_quat_w.expand(-1, self.object.num_objects, -1),
            self.object.data.object_pos_w,
            self.object.data.object_quat_w,
        )

        list_grasping_reference = []

        for env_id in range(env.num_envs):
            ee_quat_ref_target = quat_mul(
                self.ee_quat_reference_obj[
                    env.rigid_objects_in_focus[env_id][0]
                ].squeeze(),
                object_quat_ee[env_id, env.rigid_objects_in_focus[env_id][0]].squeeze(),
            )

            list_grasping_reference.append(ee_quat_ref_target)

        tensor_quat_grasp_ref = torch.stack(list_grasping_reference)

        # For Debug
        # o3d_viz_body_key_points_obj_frames(
        #     bodies_pos_ee[env_id],
        #     ee_key_points_ref,
        #     "/home/yefim-home/Downloads/Telegram Desktop/dataset/power_drills/model_1/object_convex_decomposition_meter_unit.obj",
        #     object_pos_ee[env.rigid_objects_in_focus[env_id][0], 0],
        #     object_quat_ee[env.rigid_objects_in_focus[env_id][0], 0]
        # )
        return tensor_quat_grasp_ref


class frame_in_init_ee_frame(ManagerTermBase):

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        """Class observation term to compute the current end-effector frame in the initial end-effector frame, which defined in initial_ee_frame_root.
        The first step of the environment is not used to compute the initial end-effector frame, because the robot configuration equal to the configuration in usd file.
        """
        self.initial_ee_pos_root = (
            torch.Tensor(cfg.params["initial_ee_frame_root"]["pos"])
            .unsqueeze(0)
            .repeat(env.num_envs, 1)
            .to(env.device)
        )
        self.initial_ee_quat_root = (
            torch.Tensor(cfg.params["initial_ee_frame_root"]["quat"])
            .unsqueeze(0)
            .repeat(env.num_envs, 1)
            .to(env.device)
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        initial_ee_frame_root: dict,
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        """Compute the current end-effector frame in the initial end-effector frame, which defined in initial_ee_frame_root.
        Args:
            env (ManagerBasedRLEnv): The environment instance.
            initial_ee_frame_root (dict): The initial end-effector frame in the root frame. The dictionary should contain "pos" and "quat" keys.
            ee_frame_cfg (SceneEntityCfg, optional): The configuration for the end-effector frame. Defaults to SceneEntityCfg("ee_frame").
        Returns:
            torch.Tensor: The current end-effector frame in the initial end-effector frame.
        """
        ee_frame: RigidObject = env.scene[ee_frame_cfg.name]
        ee_pos_b = ee_frame.data.target_pos_source[:, 0]
        ee_quat_b = ee_frame.data.target_quat_source[:, 0]

        ee_pos_init, ee_quat_init = subtract_frame_transforms(
            self.initial_ee_pos_root, self.initial_ee_quat_root, ee_pos_b, ee_quat_b
        )

        ee_frame_init = torch.cat([ee_pos_init, ee_quat_init], dim=-1)
        return ee_frame_init


def generated_commands_rel_frame(
    env: ManagerBasedRLEnv, command_name: str, frame_cfg
) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    ee_frame: RigidObject = env.scene[frame_cfg.name]
    ee_pos_b = ee_frame.data.target_pos_source[:, 0]
    ee_quat_b = ee_frame.data.target_quat_source[:, 0]
    command_frame = env.command_manager.get_command(command_name)
    
    command_pos_ee, command_quat_ee = subtract_frame_transforms(
        ee_pos_b, ee_quat_b,
        command_frame[:,:3], command_frame[:,3:]
    )
    
    command_frame_ee = torch.cat([command_pos_ee, command_quat_ee], dim=-1)

    return command_pos_ee #command_frame_ee


# ====================================
# ======= Forces Sensors =============
# ====================================


def binary_contact(
    env: ManagerBasedRLEnv, 
    thumb_rot_cfgs: SceneEntityCfg,
    thumb_flex_cfgs: SceneEntityCfg,
    thumb_finray_cfgs: SceneEntityCfg,
    right_flex_cfgs: SceneEntityCfg,
    right_finray_cfgs: SceneEntityCfg,
    left_flex_cfgs: SceneEntityCfg,
    left_finray_cfgs: SceneEntityCfg,
    threshold,
) -> torch.Tensor:
    """"""
    # extract the used quantities (to enable type-hinting)
    thumb_sensors: list[ContactSensor] = [
        env.scene.sensors[thumb_cfg.name]
        for thumb_cfg in [thumb_rot_cfgs, thumb_flex_cfgs, thumb_finray_cfgs]
    ]
    right_sensor: list[ContactSensor] = [
        env.scene.sensors[right_cfg.name]
        for right_cfg in [right_flex_cfgs, right_finray_cfgs]
    ]
    left_sensor: list[ContactSensor] = [
        env.scene.sensors[left_cfg.name]
        for left_cfg in [left_flex_cfgs, left_finray_cfgs]
    ]
    # check if contact force is above threshold
    contact_thumb = torch.cat(
            [
                torch.norm(sensor.data.force_matrix_w[:, :, 0], dim=-1) > threshold
                for sensor in thumb_sensors
            ],
            dim=-1,
            )
    
    contact_right = torch.cat(
            [
                torch.norm(sensor.data.force_matrix_w[:, :, 0], dim=-1) > threshold
                for sensor in right_sensor
            ],
            dim=-1)
    
    contact_left = torch.cat(
            [
                torch.norm(sensor.data.force_matrix_w[:, :, 0], dim=-1) > threshold
                for sensor in left_sensor
            ],
            dim=-1,
        )

    # sum over contacts for each environment
    res = torch.cat([contact_thumb, contact_right, contact_left], dim=-1)
    return res
    
def contact_force(
    env: ManagerBasedRLEnv,
    thumb_rot_cfgs: SceneEntityCfg,
    thumb_flex_cfgs: SceneEntityCfg,
    thumb_finray_cfgs: SceneEntityCfg,
    right_flex_cfgs: SceneEntityCfg,
    right_finray_cfgs: SceneEntityCfg,
    left_flex_cfgs: SceneEntityCfg,
    left_finray_cfgs: SceneEntityCfg,
    threshold,
    ) -> torch.Tensor:
    
    """"""
    # extract the used quantities (to enable type-hinting)
    thumb_sensors: list[ContactSensor] = [
        env.scene.sensors[thumb_cfg.name]
        for thumb_cfg in [thumb_rot_cfgs, thumb_flex_cfgs, thumb_finray_cfgs]
    ]
    right_sensor: list[ContactSensor] = [
        env.scene.sensors[right_cfg.name]
        for right_cfg in [right_flex_cfgs, right_finray_cfgs]
    ]
    left_sensor: list[ContactSensor] = [
        env.scene.sensors[left_cfg.name]
        for left_cfg in [left_flex_cfgs, left_finray_cfgs]
    ]
    # check if contact force is above threshold
    contact_force_thumb = torch.cat(
            [
                torch.norm(sensor.data.force_matrix_w[:, :, 0], dim=-1)
                for sensor in thumb_sensors
            ],
            dim=-1,
        )
    contact_force_right = torch.cat(
            [
                torch.norm(sensor.data.force_matrix_w[:, :, 0], dim=-1)
                for sensor in right_sensor
            ],
            dim=-1,
        )
    contact_force_left = torch.cat(
            [
                torch.norm(sensor.data.force_matrix_w[:, :, 0], dim=-1)
                for sensor in left_sensor
            ],
            dim=-1,
        )
    
    # sum over contacts for each environment
    contact_forces_finger = torch.cat(
            [contact_force_thumb, contact_force_right, contact_force_left],
            dim=-1,
        )
    
    return contact_forces_finger