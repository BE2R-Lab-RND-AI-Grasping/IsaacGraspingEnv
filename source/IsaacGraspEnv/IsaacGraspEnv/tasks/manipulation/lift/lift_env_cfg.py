# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

import numpy as np
from scipy.spatial.transform import Rotation as R

##
# Scene definition
##

@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    
    thumb_ft_frame: FrameTransformerCfg = MISSING
    right_ft_frame: FrameTransformerCfg = MISSING
    left_ft_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.8, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
                         semantic_tags = [("class","table"), ("color", "gray")],),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )
    contact_forces_thumb_rot = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/Link_thumb_rotation", filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],update_period=0.0, history_length=6)
    contact_forces_thumb_flex = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/Link_thumb_flexion", filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],update_period=0.0, history_length=6)
    contact_forces_thumb_finray = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/Link_thumb_finray_proxy", filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],update_period=0.0, history_length=6)
    contact_forces_right_flex = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/Link_right_flexion", filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],update_period=0.0, history_length=6)
    contact_forces_right_finray = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/Link_right_finray_proxy", filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],update_period=0.0, history_length=6)
    contact_forces_left_flex = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/Link_left_flexion", filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],update_period=0.0, history_length=6)
    contact_forces_left_finray = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/Link_left_finray_proxy", filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],update_period=0.0, history_length=6)
    contact_forces_arm = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/lbr_iiwa_link_[1-7]",update_period=0.0, history_length=6)
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # Grasp from front
            pos_x=(0.9, 1.0), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
            # Grasp from top
            # pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg | mdp.JointPositionActionCfg | mdp.JointPositionToLimitsActionCfg | mdp.EMAJointPositionToLimitsActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        ee_frame = ObsTerm(func=mdp.frame_in_robot_root_frame)
        # joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized, params={"asset_cfg":SceneEntityCfg("robot", joint_names=["Joint_right_abduction", "Joint_right_dynamixel_crank",
        #                                                                                                      "Joint_left_abduction", "Joint_left_dynamixel_crank",
        #                                                                                                      "Joint_thumb_rotation", "Joint_thumb_abduction", "Joint_thumb_dynamixel_crank"])})
        fingertips_positions = ObsTerm(func=mdp.pos_fingertips_root_frame) 
        
        # joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized)
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        
        joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized, params={"asset_cfg":SceneEntityCfg("robot", joint_names=["lbr_.*",
                                                                                                                          "Joint_.*_abduction", "Joint_.*_dynamixel_crank", "Joint_.*_rotation",
                                                                                                                          "Joint_.*_flexion", "Joint_.*_finray_proxy"])})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg":SceneEntityCfg("robot", joint_names=["lbr_.*",
                                                                                                            "Joint_.*_abduction", "Joint_.*_dynamixel_crank", "Joint_.*_rotation",
                                                                                                            "Joint_.*_flexion", "Joint_.*_finray_proxy"])})
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        object_quat = ObsTerm(func=mdp.object_quat_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # reset_object_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("object", body_names="Object"),
    #     },
    # )

is_contact_params = {"thumb_rot_cfgs": SceneEntityCfg("contact_forces_thumb_rot"),
            "thumb_flex_cfgs": SceneEntityCfg("contact_forces_thumb_flex"),
            "thumb_finray_cfgs": SceneEntityCfg("contact_forces_thumb_finray"),
            "right_flex_cfgs": SceneEntityCfg("contact_forces_right_flex"),
            "right_finray_cfgs": SceneEntityCfg("contact_forces_right_finray"),
            "left_flex_cfgs": SceneEntityCfg("contact_forces_left_flex"),
            "left_finray_cfgs": SceneEntityCfg("contact_forces_left_finray"), "threshold": 1.0}

add_is_contact_param = lambda b: b.update(is_contact_params) or b

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    fingettips_to_object = RewTerm(func=mdp.object_fingertips_distance, params={"std": 0.06}, weight=1.0)

    # lifting_object = RewTerm(func=mdp.object_is_lifted, params=add_is_contact_param({"minimal_height": 0.04}), weight=15.0)
    lifting_object = RewTerm(func=mdp.object_lift, params=add_is_contact_param({"minimal_height": 0.2}), weight=10.0)
    lifted_object = RewTerm(func=mdp.object_is_lifted, params=add_is_contact_param({"minimal_height": 0.02}), weight=1.0)
    

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params=add_is_contact_param({"std": 0.3, "minimal_height": 0.2, "command_name": "object_pose"}),
        weight=1.0,
    )

    # object_goal_tracking_fine_grained = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params=add_is_contact_param({"std": 0.05, "minimal_height": 0.2, "command_name": "object_pose"}),
    #     weight=5.0,
    # )
    hand_object_contact = RewTerm(
        func=mdp.object_hand_contact,
        weight=0.5,
        params=is_contact_params,
    )
    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2_clip,
        weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    contact_penalty = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1e-1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces_arm", body_names="lbr_.*"), "threshold": 1.0},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    spring_offset = CurrTerm(
        func=mdp.set_pd_offset,
        params={"robot_cfg":SceneEntityCfg("robot", joint_names=["Joint_.*_finray_proxy"]), "pd_offset":-0.62}
        )
    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-3, "num_steps": 30000}
    # )

#     joint_vel = CurrTerm(
#         func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-2, "num_steps": 30000}
#     )


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 8.0
        # simulation settings
        self.sim.dt = 0.01 #1.0 /120.0  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        # self.sim.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
        # self.sim.physx.gpu_max_rigid_contact_count=2**20
        # self.sim.physx.gpu_max_rigid_patch_count=2**23
        # self.sim.physx.num_threads = 4
        # self.sim.physx.solver_type = 1  # 0: pgs, 1: tgs
        # self.sim.physx.num_position_iterations = 4 # 8 bottle
        # self.sim.physx.num_velocity_iterations = 0
        # self.sim.physx.contact_offset = 0.002
        # self.sim.physx.rest_offset = 0.0
        # self.sim.physx.bounce_threshold_velocity = 0.2
        # self.sim.physx.max_depenetration_velocity = 1000.0
        # self.sim.physx.default_buffer_size_multiplier = 5.0

        self.sim.physx.max_position_iteration_count=192  # Important to avoid interpenetration.
        self.sim.physx.max_velocity_iteration_count=1
        self.sim.physx.bounce_threshold_velocity=0.2
        self.sim.physx.friction_offset_threshold=0.01
        self.sim.physx.friction_correlation_distance=0.00625
        self.sim.physx.gpu_max_rigid_contact_count=2**23
        self.sim.physx.gpu_max_rigid_patch_count=2**23
        self.sim.physx.gpu_max_num_partitions=1  # Important for stable simulation.