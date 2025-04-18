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



# Old value
# CAM_POS = np.array([1.93, -0.3553, 0.35891])
# CAM_ROT = R.from_quat([0.39126, -0.1423, 0.0799, 0.90374], scalar_first=True)
# Upd 13.03.25
CAM_POS = np.array([1.955, -1.29826, 0.64681])
CAM_ROT = R.from_quat([-0.1423, 0.0799, 0.90374, 0.39126])

##
# Scene definition
##

@configclass
class ObjectCamTableSceneCfg(ObjectTableSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """
    tiled_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/front_cam",
        update_period=0.1,
        # height=80,
        height=100,
        # width=80,
        width=100,
        # height=720,
        # width=720,
        colorize_semantic_segmentation=False,
        semantic_filter="class: object | robot | table",
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation", "distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            # focal_length=1.93, focus_distance=0.6, horizontal_aperture=3.896, vertical_aperture=2.453, clipping_range=(0.1, 1.0e5)
            # focal_length=19.3, focus_distance=5.6, horizontal_aperture=38.96, vertical_aperture=245.3, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=tuple(CAM_POS.tolist()),
                                   rot=tuple(CAM_ROT.as_quat()[[3,0,1,2]].tolist()),
                                   convention="world"),
    )

@configclass
class ProprioceptionRobotObservation(ObsGroup):
        ee_frame = ObsTerm(func=mdp.frame_in_robot_root_frame)
        fingertips_positions = ObsTerm(func=mdp.pos_fingertips_root_frame) 

        joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized, params={"asset_cfg":SceneEntityCfg("robot", joint_names=["lbr_.*",
                                                                                                                          "Joint_.*_abduction", "Joint_.*_dynamixel_crank", "Joint_.*_rotation",
                                                                                                                          "Joint_.*_flexion", "Joint_.*_finray_proxy"])})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg":SceneEntityCfg("robot", joint_names=["lbr_.*",
                                                                                                            "Joint_.*_abduction", "Joint_.*_dynamixel_crank", "Joint_.*_rotation",
                                                                                                            "Joint_.*_flexion", "Joint_.*_finray_proxy"])})

        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)


@configclass
class PointCloudObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PointCloudPolicyCfg(ProprioceptionRobotObservation):
        """Observations for policy group."""

        # image = ObsTerm(
        #     func = mdp.point_cloud_feature_extractor,
        #     params={"sensor_cfg":SceneEntityCfg("tiled_camera"),
        #             "pc_feature_extractor": mdp.PointNetMLP(device="cpu"),
        #             "data_type":"distance_to_camera",
        #             "class_filter":["robot", "object"]}
        # )
        
        point_cloud = ObsTerm(
            func = mdp.depth2point_cloud,
            params={"sensor_cfg":SceneEntityCfg("tiled_camera"),
                    "data_type":"distance_to_camera",
                    "class_filter":["robot", "object"]}
        )
        
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PointCloudPolicyCfg = PointCloudPolicyCfg()


@configclass
class FullObjPCObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class FullObjPCPolicyCfg(ProprioceptionRobotObservation):
        """Observations for policy group."""

        point_cloud = ObsTerm(
            func = mdp.full_obj_point_cloud,
            params = {
                "object_cfg": SceneEntityCfg("object"),
                "robot_cfg": SceneEntityCfg("robot"),
                "path_to_point_cloud": "/home/yefim-home/Downloads/Telegram Desktop/dataset/power_drills/model_1/point_cloud_labeled.ply",
                "num_pc": 100
                
            }
        )

        
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: FullObjPCPolicyCfg = FullObjPCPolicyCfg()

##
# Environment configuration
##


@configclass
class PointCloudLiftEnvCfg(LiftEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectCamTableSceneCfg = ObjectCamTableSceneCfg(num_envs=4096, env_spacing=2)
    # Basic settings
    observations: PointCloudObservationsCfg = PointCloudObservationsCfg()
    
    
@configclass
class FullObjPCLiftEnvCfg(LiftEnvCfg):
    """Configuration for the lifting environment."""

    # Basic settings
    observations: FullObjPCObservationsCfg = FullObjPCObservationsCfg()