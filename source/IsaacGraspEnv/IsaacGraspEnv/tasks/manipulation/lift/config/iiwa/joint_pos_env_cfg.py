# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from IsaacGraspEnv.tasks.manipulation.lift.lift_env_cfg import LiftEnvCfg
from IsaacGraspEnv.tasks.manipulation.lift.lift_cam_env_cfg import CamLiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from IsaacGraspEnv.robots.iiwa_cringe.iiwa_cringe_cfg import IIWA_CRINGE_CFG  # isort: skip

import numpy as np
from scipy.spatial.transform import Rotation as R

# OBJ_POS = np.array([0.97163, 0.04869, 0.07077])
OBJ_POS = np.array([0.9, 0.04869, 0.07077])
OBJ_ROT = R.from_euler('xyz', [90.0, 0.0, 180.0],  degrees=True)

@configclass
class IiwaCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = IIWA_CRINGE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["lbr_iiwa_joint_.*"], scale=0.5, use_default_offset=True
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
            joint_names=["Joint_.*_abduction", "Joint_.*_dynamixel_crank", "Joint_.*_rotation"],
            rescale_to_limits = True,
            # scale=1.0
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "lbr_iiwa_link_7"

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            # From Front
            # init_state=RigidObjectCfg.InitialStateCfg(pos=[0.9, 0, 0.08], rot=[0.    , 0.    , 0.7071, 0.7071]),
            init_state=RigidObjectCfg.InitialStateCfg(pos=OBJ_POS.tolist(), rot=OBJ_ROT.as_quat()[[3,0,1,2]].tolist()),
            # From Top
            # init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                # usd_path=f"source/IsaacGraspEnv/IsaacGraspEnv/assets/data/YCB/035_power_drill_wo_texture.usd",
                usd_path=f"source/IsaacGraspEnv/IsaacGraspEnv/assets/data/YCB/035_power_drill_wo_texture_light.usd",
                # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/035_power_drill.usd",
                # usd_path=f"source/IsaacGraspEnv/IsaacGraspEnv/assets/data/YCB/Props/instanceable_meshes.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=3666.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=192,
                    solver_velocity_iteration_count=1,
                    max_contact_impulse=1e32,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                semantic_tags = [("class","object"), ("color", "red")],
            ),
            # Grasp From Front
            # spawn=sim_utils.CylinderCfg(
            # radius=0.028,
            # height=0.14,
            # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0,0.0,0.0)),
            # mass_props=sim_utils.MassPropertiesCfg(density=1000),
            # rigid_props=sim_utils.RigidBodyPropertiesCfg(
            #         # solver_position_iteration_count=16,
            #         # solver_velocity_iteration_count=1,
            #         # max_angular_velocity=1000.0,
            #         # max_linear_velocity=1000.0,
            #         # max_depenetration_velocity=5.0,
            #         # disable_gravity=False,
                    
            #     # kinematic_enabled=False,
            #     # disable_gravity=False,
            #     # enable_gyroscopic_forces=True,
            #     # solver_position_iteration_count=8,
            #     # solver_velocity_iteration_count=0,
            #     # sleep_threshold=0.005,
            #     # stabilization_threshold=0.0025,
            #     # max_depenetration_velocity=1000.0,
                
            #     disable_gravity=False,
            #     max_depenetration_velocity=5.0,
            #     linear_damping=0.0,
            #     angular_damping=0.0,
            #     max_linear_velocity=1000.0,
            #     max_angular_velocity=3666.0,
            #     enable_gyroscopic_forces=True,
            #     solver_position_iteration_count=192,
            #     solver_velocity_iteration_count=1,
            #     max_contact_impulse=1e32,
            #         ),
            # physics_material=sim_utils.RigidBodyMaterialCfg(            
            #     static_friction=1.0,
            #     dynamic_friction=1.0,),
            # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            # ),
            # Grasp From Top
            # spawn=sim_utils.CuboidCfg(
            # size=[0.06,0.06,0.06],
            # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0,0.0,0.0)),
            # mass_props=sim_utils.MassPropertiesCfg(density=1000),
            # rigid_props=sim_utils.RigidBodyPropertiesCfg(
            #         # solver_position_iteration_count=16,
            #         # solver_velocity_iteration_count=1,
            #         # max_angular_velocity=1000.0,
            #         # max_linear_velocity=1000.0,
            #         # max_depenetration_velocity=5.0,
            #         # disable_gravity=False,
                    
            #     # kinematic_enabled=False,
            #     # disable_gravity=False,
            #     # enable_gyroscopic_forces=True,
            #     # solver_position_iteration_count=8,
            #     # solver_velocity_iteration_count=0,
            #     # sleep_threshold=0.005,
            #     # stabilization_threshold=0.0025,
            #     # max_depenetration_velocity=1000.0,
                
            #     disable_gravity=False,
            #     max_depenetration_velocity=5.0,
            #     linear_damping=0.0,
            #     angular_damping=0.0,
            #     max_linear_velocity=1000.0,
            #     max_angular_velocity=3666.0,
            #     enable_gyroscopic_forces=True,
            #     solver_position_iteration_count=192,
            #     solver_velocity_iteration_count=1,
            #     max_contact_impulse=1e32,
            #         ),
            # physics_material=sim_utils.RigidBodyMaterialCfg(            
            #     static_friction=1.0,
            #     dynamic_friction=1.0,),
            # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            # ),
        )

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
class IiwaCubeCamLiftEnvCfg(CamLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = IIWA_CRINGE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["lbr_iiwa_joint_.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.JointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=["Joint_.*_abduction", "Joint_.*_dynamixel_crank", "Joint_.*_rotation"],
            rescale_to_limits = True,
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "lbr_iiwa_link_7"

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            # From Front
            init_state=RigidObjectCfg.InitialStateCfg(pos=OBJ_POS.tolist(), rot=OBJ_ROT.as_quat()[[3,0,1,2]].tolist()),
            spawn=UsdFileCfg(
                usd_path=f"source/IsaacGraspEnv/IsaacGraspEnv/assets/data/YCB/035_power_drill.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=3666.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=192,
                    solver_velocity_iteration_count=1,
                    max_contact_impulse=1e32,
                        ),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                semantic_tags = [("class","object"), ("color", "red")],
                ),
            # ),
            # Grasp From Front
            # spawn=sim_utils.CylinderCfg(
            # radius=0.028,
            # height=0.14,
            # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0,0.0,0.0)),
            # mass_props=sim_utils.MassPropertiesCfg(density=1000),
            # rigid_props=sim_utils.RigidBodyPropertiesCfg(
            #         # solver_position_iteration_count=16,
            #         # solver_velocity_iteration_count=1,
            #         # max_angular_velocity=1000.0,
            #         # max_linear_velocity=1000.0,
            #         # max_depenetration_velocity=5.0,
            #         # disable_gravity=False,
                    
            #     # kinematic_enabled=False,
            #     # disable_gravity=False,
            #     # enable_gyroscopic_forces=True,
            #     # solver_position_iteration_count=8,
            #     # solver_velocity_iteration_count=0,
            #     # sleep_threshold=0.005,
            #     # stabilization_threshold=0.0025,
            #     # max_depenetration_velocity=1000.0,
                
            #     disable_gravity=False,
            #     max_depenetration_velocity=5.0,
            #     linear_damping=0.0,
            #     angular_damping=0.0,
            #     max_linear_velocity=1000.0,
            #     max_angular_velocity=3666.0,
            #     enable_gyroscopic_forces=True,
            #     solver_position_iteration_count=192,
            #     solver_velocity_iteration_count=1,
            #     max_contact_impulse=1e32,
            #         ),
            # physics_material=sim_utils.RigidBodyMaterialCfg(            
            #     static_friction=1.0,
            #     dynamic_friction=1.0,),
            # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            # ),
        )

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
class IiwaCubeCamLiftEnvCfg_PLAY(IiwaCubeCamLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
