# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg, RewardTermCfg, ManagerTermBase
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .observations import (
    instance_randomize_obj_positions_in_robot_world_frame as get_obj_pos_w,
    instance_randomize_obj_vel_in_world_frame as get_obj_vel_w,
)


def object_hand_contact(
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
    is_contact_thumb = torch.sum(
        torch.cat(
            [
                torch.norm(sensor.data.force_matrix_w[:, :, 0], dim=-1) > threshold
                for sensor in thumb_sensors
            ],
            dim=-1,
        ),
        dim=1,
        keepdim=True,
    )
    is_contact_right = torch.sum(
        torch.cat(
            [
                torch.norm(sensor.data.force_matrix_w[:, :, 0], dim=-1) > threshold
                for sensor in right_sensor
            ],
            dim=-1,
        ),
        dim=1,
        keepdim=True,
    )
    is_contact_left = torch.sum(
        torch.cat(
            [
                torch.norm(sensor.data.force_matrix_w[:, :, 0], dim=-1) > threshold
                for sensor in left_sensor
            ],
            dim=-1,
        ),
        dim=1,
        keepdim=True,
    )

    # sum over contacts for each environment
    res = torch.sum(
        torch.logical_and(
            torch.logical_or(is_contact_right, is_contact_left), is_contact_thumb
        ),
        dim=1,
    )
    return res


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    thumb_rot_cfgs: SceneEntityCfg,
    thumb_flex_cfgs: SceneEntityCfg,
    thumb_finray_cfgs: SceneEntityCfg,
    right_flex_cfgs: SceneEntityCfg,
    right_finray_cfgs: SceneEntityCfg,
    left_flex_cfgs: SceneEntityCfg,
    left_finray_cfgs: SceneEntityCfg,
    threshold,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:

    is_contact = object_hand_contact(
        env,
        thumb_rot_cfgs,
        thumb_flex_cfgs,
        thumb_finray_cfgs,
        right_flex_cfgs,
        right_finray_cfgs,
        left_flex_cfgs,
        left_finray_cfgs,
        threshold,
    )
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return is_contact.float() * torch.where(
        object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0
    )


def instance_randomize_object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    thumb_rot_cfgs: SceneEntityCfg,
    thumb_flex_cfgs: SceneEntityCfg,
    thumb_finray_cfgs: SceneEntityCfg,
    right_flex_cfgs: SceneEntityCfg,
    right_finray_cfgs: SceneEntityCfg,
    left_flex_cfgs: SceneEntityCfg,
    left_finray_cfgs: SceneEntityCfg,
    threshold,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 1), fill_value=-1)

    is_contact = object_hand_contact(
        env,
        thumb_rot_cfgs,
        thumb_flex_cfgs,
        thumb_finray_cfgs,
        right_flex_cfgs,
        right_finray_cfgs,
        left_flex_cfgs,
        left_finray_cfgs,
        threshold,
    )

    """Reward the agent for lifting the object above the minimal height."""

    obj_pos_w = get_obj_pos_w(env, object_cfg)

    return is_contact.float() * torch.where(obj_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_lift(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    thumb_rot_cfgs: SceneEntityCfg,
    thumb_flex_cfgs: SceneEntityCfg,
    thumb_finray_cfgs: SceneEntityCfg,
    right_flex_cfgs: SceneEntityCfg,
    right_finray_cfgs: SceneEntityCfg,
    left_flex_cfgs: SceneEntityCfg,
    left_finray_cfgs: SceneEntityCfg,
    threshold,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:

    is_contact = object_hand_contact(
        env,
        thumb_rot_cfgs,
        thumb_flex_cfgs,
        thumb_finray_cfgs,
        right_flex_cfgs,
        right_finray_cfgs,
        left_flex_cfgs,
        left_finray_cfgs,
        threshold,
    )
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    lift = torch.clip(object.data.root_pos_w[:, 2], 0.0, minimal_height)

    return is_contact.float() * lift


def instance_randomize_object_lift(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    thumb_rot_cfgs: SceneEntityCfg,
    thumb_flex_cfgs: SceneEntityCfg,
    thumb_finray_cfgs: SceneEntityCfg,
    right_flex_cfgs: SceneEntityCfg,
    right_finray_cfgs: SceneEntityCfg,
    left_flex_cfgs: SceneEntityCfg,
    left_finray_cfgs: SceneEntityCfg,
    threshold,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 1), fill_value=-1)

    is_contact = object_hand_contact(
        env,
        thumb_rot_cfgs,
        thumb_flex_cfgs,
        thumb_finray_cfgs,
        right_flex_cfgs,
        right_finray_cfgs,
        left_flex_cfgs,
        left_finray_cfgs,
        threshold,
    )
    """Reward the agent for lifting the object above the minimal height."""

    obj_pos_w = get_obj_pos_w(env, object_cfg)

    lift = torch.clip(obj_pos_w[:, 2], 0.0, minimal_height)

    return is_contact.float() * lift


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    obj_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(obj_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def instance_randomize_object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 1), fill_value=-1)

    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    obj_pos_w = get_obj_pos_w(env, object_cfg)
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(obj_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_fingertips_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    thumb_ft_frame_cfg: SceneEntityCfg = SceneEntityCfg("thumb_ft_frame"),
    right_ft_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ft_frame"),
    left_ft_frame_cfg: SceneEntityCfg = SceneEntityCfg("left_ft_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    thumb_ft_frame: FrameTransformer = env.scene[thumb_ft_frame_cfg.name]
    right_ft_frame: FrameTransformer = env.scene[right_ft_frame_cfg.name]
    left_ft_frame: FrameTransformer = env.scene[left_ft_frame_cfg.name]
    # Target object position: (num_envs, 3)
    object_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    thumb_pos_w = thumb_ft_frame.data.target_pos_w[..., 0, :]
    right_pos_w = right_ft_frame.data.target_pos_w[..., 0, :]
    left_pos_w = left_ft_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_fingertips_distance = torch.cat(
        [
            torch.norm(object_pos_w - thumb_pos_w, dim=1, keepdim=True),
            torch.norm(object_pos_w - right_pos_w, dim=1, keepdim=True),
            torch.norm(object_pos_w - left_pos_w, dim=1, keepdim=True),
        ],
        dim=-1,
    )

    object_fingertips_distance = torch.clip(object_fingertips_distance, 0.03, 0.8)

    reward_scale = torch.tensor([0.04, 0.02, 0.02]).to(env.sim.device)

    return torch.sum(
        torch.mul(1 / (std + object_fingertips_distance), reward_scale), dim=1
    )


def instance_randomize_object_fingertips_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    thumb_ft_frame_cfg: SceneEntityCfg = SceneEntityCfg("thumb_ft_frame"),
    right_ft_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ft_frame"),
    left_ft_frame_cfg: SceneEntityCfg = SceneEntityCfg("left_ft_frame"),
) -> torch.Tensor:
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 1), fill_value=-1)

    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    thumb_ft_frame: FrameTransformer = env.scene[thumb_ft_frame_cfg.name]
    right_ft_frame: FrameTransformer = env.scene[right_ft_frame_cfg.name]
    left_ft_frame: FrameTransformer = env.scene[left_ft_frame_cfg.name]
    # Target object position: (num_envs, 3)
    object_pos_w = get_obj_pos_w(env, object_cfg)
    # End-effector position: (num_envs, 3)
    thumb_pos_w = thumb_ft_frame.data.target_pos_w[..., 0, :]
    right_pos_w = right_ft_frame.data.target_pos_w[..., 0, :]
    left_pos_w = left_ft_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_fingertips_distance = torch.cat(
        [
            torch.norm(object_pos_w - thumb_pos_w, dim=1, keepdim=True),
            torch.norm(object_pos_w - right_pos_w, dim=1, keepdim=True),
            torch.norm(object_pos_w - left_pos_w, dim=1, keepdim=True),
        ],
        dim=-1,
    )

    object_fingertips_distance = torch.clip(object_fingertips_distance, 0.03, 0.8)

    reward_scale = torch.tensor([0.04, 0.02, 0.02]).to(env.sim.device)

    return torch.sum(
        torch.mul(1 / (std + object_fingertips_distance), reward_scale), dim=1
    )


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    thumb_rot_cfgs: SceneEntityCfg,
    thumb_flex_cfgs: SceneEntityCfg,
    thumb_finray_cfgs: SceneEntityCfg,
    right_flex_cfgs: SceneEntityCfg,
    right_finray_cfgs: SceneEntityCfg,
    left_flex_cfgs: SceneEntityCfg,
    left_finray_cfgs: SceneEntityCfg,
    threshold,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:

    obj_is_lifted = object_is_lifted(
        env,
        minimal_height,
        thumb_rot_cfgs,
        thumb_flex_cfgs,
        thumb_finray_cfgs,
        right_flex_cfgs,
        right_finray_cfgs,  # type: ignore
        left_flex_cfgs,
        left_finray_cfgs,
        threshold,
    )
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b
    )
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold

    # reward = (1 - torch.tanh(distance / std)).unsqueeze(1)
    reward = 1.0 / (std + distance)

    return obj_is_lifted.float() * reward


def instance_object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    thumb_rot_cfgs: SceneEntityCfg,
    thumb_flex_cfgs: SceneEntityCfg,
    thumb_finray_cfgs: SceneEntityCfg,
    right_flex_cfgs: SceneEntityCfg,
    right_finray_cfgs: SceneEntityCfg,
    left_flex_cfgs: SceneEntityCfg,
    left_finray_cfgs: SceneEntityCfg,
    threshold,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:

    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 1), fill_value=-1)

    obj_is_lifted = instance_randomize_object_is_lifted(
        env,
        minimal_height,
        thumb_rot_cfgs,
        thumb_flex_cfgs,
        thumb_finray_cfgs,
        right_flex_cfgs,
        right_finray_cfgs,  # type: ignore
        left_flex_cfgs,
        left_finray_cfgs,
        threshold,
    )
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b
    )

    obj_pos_w = get_obj_pos_w(env, object_cfg)

    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - obj_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold

    # reward = (1 - torch.tanh(distance / std)).unsqueeze(1)
    reward = 1.0 / (std + distance)

    return obj_is_lifted.float() * reward


def instance_object_reached_target(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    threshold_reach: float,
    thumb_rot_cfgs: SceneEntityCfg,
    thumb_flex_cfgs: SceneEntityCfg,
    thumb_finray_cfgs: SceneEntityCfg,
    right_flex_cfgs: SceneEntityCfg,
    right_finray_cfgs: SceneEntityCfg,
    left_flex_cfgs: SceneEntityCfg,
    left_finray_cfgs: SceneEntityCfg,
    threshold,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):

    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 1), fill_value=-1)

    obj_is_lifted = instance_randomize_object_is_lifted(
        env,
        minimal_height,
        thumb_rot_cfgs,
        thumb_flex_cfgs,
        thumb_finray_cfgs,
        right_flex_cfgs,
        right_finray_cfgs,  # type: ignore
        left_flex_cfgs,
        left_finray_cfgs,
        threshold,
    )
    
    distance_reward = instance_object_goal_distance(
        env,
        std,
        minimal_height,
        command_name,
        thumb_rot_cfgs,
        thumb_flex_cfgs,
        thumb_finray_cfgs,
        right_flex_cfgs,
        right_finray_cfgs,
        left_flex_cfgs,
        left_finray_cfgs,
        threshold,
        robot_cfg,
        object_cfg,
    )
    
    distance = 1 / distance_reward + 0.001 - std
    
    reward = torch.where(distance <= threshold_reach, 1.0, 0.0)
    
    return obj_is_lifted.float() * reward
    
    
    
    


def joint_vel_l2_clip(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.clip(
            torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), -1.0, 1.0
        ),
        dim=1,
    )


class instance_object_displacement(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):

        super().__init__(cfg, env)

        self.object_cfg = cfg.params["object_cfg"]

        self.prev_object_pos = torch.full((env.num_envs, 3), 0).to(env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Penalize joint velocities on the articulation using L2 squared kernel.

        NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
        """

        curr_obj_pos = get_obj_pos_w(env, self.object_cfg)
        # extract the used quantities (to enable type-hinting)
        reward = torch.sum(
            torch.abs(curr_obj_pos - self.prev_object_pos),
            dim=1,
        )

        self.prev_object_pos = curr_obj_pos.clone()

        return reward


def instance_object_vel_l2(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):

    object_vel_w = get_obj_vel_w(env, object_cfg)

    return torch.linalg.vector_norm(object_vel_w, dim=1)


def robot_link_vel_w_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["ee_frame"]),
):

    robot = env.scene[asset_cfg.name]
    robot_link_vel_w = robot.data.body_state_w[:, asset_cfg.body_ids[0], 7:]

    return torch.linalg.vector_norm(robot_link_vel_w, dim=1)


# def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:object_pos_w
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     # check if contact force is above threshold
#     net_contact_forces = contact_sensor.data.net_forces_w_history
#     is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
#     # sum over contacts for each environment
#     return torch.sum(is_contact, dim=1)
