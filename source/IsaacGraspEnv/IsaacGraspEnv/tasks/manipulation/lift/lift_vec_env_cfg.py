# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm

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
        ee_frame = ObsTerm(func=mdp.frame_in_robot_root_frame)
        # fingertips_positions = ObsTerm(func=mdp.pos_fingertips_root_frame) 

        # joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized, params={"asset_cfg":SceneEntityCfg("robot", joint_names=["lbr_.*",
        #                                                                                                                   "Joint_.*_abduction", "Joint_.*_dynamixel_crank", "Joint_.*_rotation",
        #                                                                                                                   "Joint_.*_flexion", "Joint_.*_finray_proxy"])})
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg":SceneEntityCfg("robot", joint_names=["lbr_.*",
        #                                                                                                     "Joint_.*_abduction", "Joint_.*_dynamixel_crank", "Joint_.*_rotation",
        #                                                                                                     "Joint_.*_flexion", "Joint_.*_finray_proxy"])})

        joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized, params={"asset_cfg":SceneEntityCfg("robot", joint_names=["lbr_.*",
                                                                                                                    "Joint_.*"])})
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg":SceneEntityCfg("robot", joint_names=["lbr_.*",
        #                                                                                                     "Joint_.*"])})
        
        # =============== For Debugging (TODO: Delite after it) ======================= 
        # object_position = ObsTerm(func=mdp.instance_randomize_obj_positions_in_robot_root_frame)
        # object_quat = ObsTerm(func=mdp.instance_randomize_obj_orientations_in_robot_root_frame)
        object_position = ObsTerm(func=mdp.instance_randomize_obj_positions_in_robot_ee_frame, params={"frame_cfg":SceneEntityCfg("ee_frame")})
        object_quat = ObsTerm(func=mdp.instance_randomize_obj_orientations_in_robot_ee_frame, params={"frame_cfg":SceneEntityCfg("ee_frame")})
        # ======================================
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)


# ========== Test TODO: REMOVE ============
# with open("/home/yefim-home/Documents/work/IsaacGraspingEnv/grasping_reference.npy", "rb") as f:
    
#     grasp_ref = np.load(f, allow_pickle=True)

@configclass
class VectorsObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class VectorsPolicyCfg(ProprioceptionRobotObservation):
        """Observations for policy group."""

        vectors = ObsTerm(
            func = mdp.instance_vectors_joint_hand_key_points,
            params = {
                "object_cfg": SceneEntityCfg("object"),
                "robot_cfg": SceneEntityCfg("robot", body_names=["Link_.*"], joint_names=["Joint_.*"]),
                "frame_cfg": SceneEntityCfg("ee_frame"),
                "grasping_reference_path": "/home/yefim-home/Documents/work/IsaacGraspingEnv/grasping_reference.npy", #grasp_ref,#torch.zeros(3,10)
                "body_frame_key": "key_points",
                "joint_position_key": "qpos"
                
            }
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: VectorsPolicyCfg = VectorsPolicyCfg()



##
# Environment configuration
##

@configclass
class VectorsLiftEnvCfg(LiftEnvCfg):
    """Configuration for the lifting environment."""

    # Basic settings
    observations: VectorsObservationsCfg = VectorsObservationsCfg()
    
    