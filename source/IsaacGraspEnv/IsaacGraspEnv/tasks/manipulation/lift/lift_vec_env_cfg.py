# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm

from isaaclab.managers import SceneEntityCfg

from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp
from .lift_env_cfg import LiftEnvCfg, ObjectTableSceneCfg
import numpy as np
from scipy.spatial.transform import Rotation as R

import torch

##
# Scene definition
##


@configclass
class ProprioceptionRobotObservation(ObsGroup):
    # ee_frame = ObsTerm(func=mdp.frame_in_robot_root_frame)
    
    ee_frame = ObsTerm(func=mdp.frame_in_init_ee_frame, params={"initial_ee_frame_root":{"pos":[0.7295, -0.0601, 0.1157], "quat":[0.7003, 0.0348, 0.7123, -0.0285]}})
    # fingertips_positions = ObsTerm(func=mdp.pos_fingertips_root_frame)

    # joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized, params={"asset_cfg":SceneEntityCfg("robot", joint_names=["lbr_.*",
    #                                                                                                                   "Joint_.*_abduction", "Joint_.*_dynamixel_crank", "Joint_.*_rotation",
    #                                                                                                                   "Joint_.*_flexion", "Joint_.*_finray_proxy"])})
    # joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg":SceneEntityCfg("robot", joint_names=["lbr_.*",
    #                                                                                                     "Joint_.*_abduction", "Joint_.*_dynamixel_crank", "Joint_.*_rotation",
    #                                                                                                     "Joint_.*_flexion", "Joint_.*_finray_proxy"])})

    joint_pos = ObsTerm(
        func=mdp.joint_pos_limit_normalized,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["lbr_.*", "Joint_.*"])
        },
    )
    # joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg":SceneEntityCfg("robot", joint_names=["lbr_.*",
    #                                                                                                     "Joint_.*"])})

    # =============== For Debugging (TODO: Delite after it) =======================
    # object_position = ObsTerm(func=mdp.instance_randomize_obj_positions_in_robot_root_frame)
    # object_quat = ObsTerm(func=mdp.instance_randomize_obj_orientations_in_robot_root_frame)
    object_position = ObsTerm(
        func=mdp.instance_randomize_obj_positions_in_robot_ee_frame,
        params={"frame_cfg": SceneEntityCfg("ee_frame")},
    )
    object_quat = ObsTerm(
        func=mdp.instance_randomize_obj_orientations_in_robot_ee_frame,
        params={"frame_cfg": SceneEntityCfg("ee_frame")},
    )
    
    object_vel = ObsTerm(
        func= mdp.instance_randomize_obj_vel_in_robot_frame,
        params={"robot_cfg": SceneEntityCfg("robot", body_names=["lbr_iiwa_link_7"]), "object_cfg": SceneEntityCfg("object")}
    )
    
    # ======================================

    relative_target_quat_current = ObsTerm(
        func=mdp.instance_target_end_effector_orientation,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "robot_cfg": SceneEntityCfg("robot", body_names=["lbr_iiwa_link_7"]),
            "frame_cfg": SceneEntityCfg("ee_frame"),
            "grasping_reference_path": "/home/yefim-home/Documents/work/IsaacGraspingEnv/grasping_reference.npy",
            "body_quat_key":  "body_quat_obj",
        },
    )

    target_object_position = ObsTerm(
        func=mdp.generated_commands_rel_frame, #generated_commands,
        params={"command_name": "object_pose", "frame_cfg":SceneEntityCfg("ee_frame")}
    )
    actions = ObsTerm(func=mdp.last_action)


@configclass
class VectorsObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class VectorsPolicyCfg(ProprioceptionRobotObservation):
        """Observations for policy group."""

        vectors = ObsTerm(
            func=mdp.instance_vectors_joint_hand_key_points,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "robot_cfg": SceneEntityCfg(
                    "robot", body_names=["Link_.*"], joint_names=["Joint_.*"]
                ),
                "frame_cfg": SceneEntityCfg("ee_frame"),
                "grasping_reference_path": "/home/yefim-home/Documents/work/IsaacGraspingEnv/grasping_reference.npy",  # grasp_ref,#torch.zeros(3,10)
                "body_frame_key": "key_points",
                "joint_position_key": "qpos",
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: VectorsPolicyCfg = VectorsPolicyCfg()


# ===========================
# Reward Configuration
# ===========================

adder_contact_sensor_params = mdp.generate_contact_sensor_params()
@configclass
class VecRewardsCfg:
    """Reward terms for the MDP."""
    
    norm_vectors = RewTerm(func=mdp.instance_vectors_norm, params={"name_obs_vector":"vectors"}, weight= -1/12*4/5*5/3 /0.2)#-5.0/0.2/10) # Weight (1) normalization ~ 1, (2) ~ value reward, (3) extened to [-1, 1]

    object_goal_tracking = RewTerm(
        func=mdp.instance_object_goal_distance,
        # For power drills
        params=adder_contact_sensor_params({"std": 0.04, "minimal_height": 0.13, "command_name": "object_pose"}),
        # for screwdrives
        # params=adder_contact_sensor_params({"std": 0.04, "minimal_height": 0.025, "command_name": "object_pose"}),
        weight=1/30 * 3/5 * 5/3 /0.2, #1.0/0.2/10, 
    )
    
    
    object_goal_reach = RewTerm(
        func=mdp.instance_object_reached_target,
        # For power drills
        params=adder_contact_sensor_params({"std": 0.04, "minimal_height": 0.13, "command_name": "object_pose"}),
        # for screwdrives
        # params=adder_contact_sensor_params({"threshold_reach":0.05, "std": 0.04, "minimal_height": 0.025, "command_name": "object_pose"}),
        weight=1/5 * 5/5 * 5/3 / 0.2, #5.0/0.2/10, 
    )
    
    hand_object_contact = RewTerm(
        func=mdp.object_hand_contact,
        weight= 1/4 * 2/5 * 5/3 /0.2, #0.75/0.2/10,
        params=adder_contact_sensor_params({}),
    )
    hand_object_contact_force = RewTerm(
        func=mdp.object_hand_force_contact,
        weight= 1/500 * 3/5 * 5/3 /0.2, #0.01/0.2/10,
        params=adder_contact_sensor_params({}),
    )
    
    
    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1/300 * 3/5 * 5/3 / 0.2)#-5e-3/0.2/10)
    
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2_clip,
        weight= -1/10 * 1/5 * 5/3 / 0.2, # -5e-3/0.2/10,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Joint.*"])},
    )
    
    ee_vel_l2 = RewTerm(
        func=mdp.robot_link_vel_w_l2,
        weight= -1/45 * 2/5 * 5/3 / 0.2, #-8e-3/0.2/10,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["lbr_iiwa_link_7"]), },
    )

    # contact_penalty = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-1e-0/0.2/10,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces_arm", body_names="lbr_.*"), "threshold": 1.0},
    # )


##
# Environment configuration
##


@configclass
class VectorsLiftEnvCfg(LiftEnvCfg):
    """Configuration for the lifting environment."""

    # Basic settings
    observations: VectorsObservationsCfg = VectorsObservationsCfg()
    rewards: VecRewardsCfg = VecRewardsCfg()
