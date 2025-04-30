# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy.spatial.transform import Rotation as R

from isaaclab.assets import RigidObjectCollectionCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from IsaacGraspEnv.tasks.manipulation.lift.lift_env_cfg import LiftEnvCfg
from IsaacGraspEnv.tasks.manipulation.lift.lift_cam_env_cfg import (
    ObjectCamTableSceneCfg,
    PointCloudObservationsCfg,
    FullObjPCObservationsCfg,
)

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from IsaacGraspEnv.robots.iiwa_cringe.iiwa_cringe_cfg import (
    IIWA_CRINGE_CFG,
)  # isort: skip

from dataclasses import MISSING

# OBJ_POS = np.array([0.97163, 0.04869, 0.07077])
# OBJ_POS = np.array([0.9, 0.04869, 0.07077])
# OBJ_ROT = R.from_euler('xyz', [90.0, 0.0, 180.0],  degrees=True)


# OBJ_POS = np.array([0.0, 0.0, 0.0])
OBJ_POS = np.array([0.9, 0.0, 0.07077])
# OBJ_POS = np.array([0.0, 0.0, 10.0])
OBJ_ROT = R.from_euler("xyz", [90.0, 0.0, 0.0], degrees=True)


@configclass
class IiwaCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = IIWA_CRINGE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["lbr_iiwa_joint_.*"],
            scale=0.5,
            use_default_offset=True,
        )
        # self.actions.gripper_action = mdp.JointPositionToLimitsActionCfg(
        #     asset_name="robot",
        #     joint_names=["Joint_.*_abduction", "Joint_.*_dynamixel_crank", "Joint_.*_rotation"],
        #     scale=1.0,
        # )
        # self.actions.gripper_action = mdp.JointPositionToLimitsActionCfg(
        #     asset_name="robot",
        #     joint_names=["Joint_.*"],
        #     rescale_to_limits = True,
        # )
        self.actions.gripper_action = mdp.JointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=[
                "Joint_.*_abduction",
                "Joint_.*_dynamixel_crank",
                "Joint_.*_rotation",
            ],
            rescale_to_limits=True,
            # scale=1.0
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "lbr_iiwa_link_7"

        # Set Cube as object
        self.scene.object = RigidObjectCollectionCfg(rigid_objects=MISSING)

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/lbr_iiwa_link_0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/lbr_iiwa_link_7",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.12],
                    ),
                ),
            ],
        )
        self.scene.thumb_ft_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/lbr_iiwa_link_0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/Link_thumb_finray_proxy",
                    name="fingertips_thumb",
                    offset=OffsetCfg(
                        pos=[0.055, 0.02, 0.0],
                    ),
                ),
            ],
        )
        self.scene.right_ft_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/lbr_iiwa_link_0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/Link_right_finray_proxy",
                    name="fingertips_right",
                    offset=OffsetCfg(
                        pos=[-0.02, 0.055, 0],
                    ),
                ),
            ],
        )
        self.scene.left_ft_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/lbr_iiwa_link_0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/Link_left_finray_proxy",
                    name="fingertips_left",
                    offset=OffsetCfg(
                        pos=[-0.02, 0.055, 0.0],
                    ),
                ),
            ],
        )


@configclass
class IiwaCubeLiftEnvCfg_PLAY(IiwaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


@configclass
class IiwaCubePCLiftEnvCfg(IiwaCubeLiftEnvCfg):

    # Scene settings
    scene: ObjectCamTableSceneCfg = ObjectCamTableSceneCfg(num_envs=4096, env_spacing=2)
    # Basic settings
    observations: PointCloudObservationsCfg = PointCloudObservationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()


@configclass
class IiwaCubePCLiftEnvCfg_PLAY(IiwaCubePCLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


@configclass
class IiwaCubeFullObjPCLiftEnvCfg(IiwaCubeLiftEnvCfg):

    # Basic settings
    observations: FullObjPCObservationsCfg = FullObjPCObservationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()


@configclass
class IiwaCubeFullObjPCLiftEnvCfg_PLAY(IiwaCubeFullObjPCLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
