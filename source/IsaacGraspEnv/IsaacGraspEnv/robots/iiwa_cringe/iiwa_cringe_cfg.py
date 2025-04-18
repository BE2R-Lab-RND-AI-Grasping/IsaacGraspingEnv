# Copyright (c) 2025-2027, BE2R Lab
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Kuka iiwa r14 with egor cringe hand robots.

The following configurations are available:

* :obj:`IIWA_CRINGE_CFG`: Kuka iiwa r14 with egor cringe hand
* :obj:`IIWA_CRINGE_HIGH_PD_CFG`: Kuka iiwa r14 with egor cringe hand with stiffer PD control
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##
from pathlib import Path
import numpy as np

# relative_path = "source/IsaacGraspEnv/IsaacGraspEnv/robots/iiwa_cringe/usd/iiwa_cringe_v2_open.usd"
# relative_path = "source/IsaacGraspEnv/IsaacGraspEnv/robots/iiwa_cringe/usd/iiwa_cringe_v1.usd"
relative_path = "source/IsaacGraspEnv/IsaacGraspEnv/robots/hand_iiwa/hand_iiwa.usd"
ABSOLUTE_PATH = Path(relative_path).resolve()

# INIT_Q_IIWA = np.array([-2.9, 71.8, 0.0, -89.3, 0.0, -71.8, 0.0])/180*np.pi
INIT_Q_IIWA = np.array([-2.9-2.1, 71.8, 0.0, -89.3, 0.0, -71.8, 0.0])/180*np.pi
INIT_Q_IIWA = INIT_Q_IIWA.tolist()

IIWA_CRINGE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ABSOLUTE_PATH.as_posix(),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # disable_gravity=False,
            # retain_accelerations=False,
            # enable_gyroscopic_forces=False,
            # angular_damping=0.01,
            # max_linear_velocity=1000.0,
            # max_angular_velocity= 130 / np.pi * 180.0,
            # max_depenetration_velocity=1000.0,
            # max_contact_impulse=1e32,
            disable_gravity=True,
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
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=0,
            # sleep_threshold=0.005,
            # stabilization_threshold=0.0005,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0015),
        semantic_tags = [("class","robot"), ("color", "orange")],
    ),
    # -0.036,  1.204,   2.9670658, 1.906,    2.9671504,  -1.555,-2.926
    init_state=ArticulationCfg.InitialStateCfg(
        # Grasp from front
        joint_pos={
            # Open Kinematick
            # "lbr_iiwa_joint_1": -0.036,
            # "lbr_iiwa_joint_2": 1.0,
            # "lbr_iiwa_joint_3": 2.966,
            # "lbr_iiwa_joint_4": 1.906,
            # "lbr_iiwa_joint_5": 2.966,
            # "lbr_iiwa_joint_6": -1.355,
            # "lbr_iiwa_joint_7": -2.926,
            
            # "lbr_iiwa_joint_1": -0.036,
            # "lbr_iiwa_joint_2": 0.6,
            # "lbr_iiwa_joint_3": 2.966,
            # "lbr_iiwa_joint_4": 1.0,
            # "lbr_iiwa_joint_5": -2.966,
            # "lbr_iiwa_joint_6": 1.57,
            # "lbr_iiwa_joint_7": -2.926,
            
            # "Joint_thumb_rotation":1.569,
            # "Joint_thumb_abduction":0.0,
            # "Joint_thumb_flexion":0.0,
            # "Joint_thumb_finray_proxy":0.0,
            # "Joint_right_abduction":0.0,
            # "Joint_right_flexion":0.0,
            # "Joint_right_finray_proxy":0.0,
            # "Joint_left_abduction":0.0,
            # "Joint_left_flexion":0.0,
            # "Joint_left_finray_proxy":0.0,
            
            # Close to object
            # "lbr_iiwa_joint_1": 0.0,
            # "lbr_iiwa_joint_2": 1.4,
            # "lbr_iiwa_joint_3": 3.14,
            # "lbr_iiwa_joint_4": 1.6,
            # "lbr_iiwa_joint_5": -3.466,
            # "lbr_iiwa_joint_6": -1.155,
            # "lbr_iiwa_joint_7": -2.926,
            
            "lbr_iiwa_joint_1": INIT_Q_IIWA[0],
            "lbr_iiwa_joint_2": INIT_Q_IIWA[1],
            "lbr_iiwa_joint_3": INIT_Q_IIWA[2],
            "lbr_iiwa_joint_4": INIT_Q_IIWA[3],
            "lbr_iiwa_joint_5": INIT_Q_IIWA[4],
            "lbr_iiwa_joint_6": INIT_Q_IIWA[5],
            "lbr_iiwa_joint_7": INIT_Q_IIWA[6],
            
            "Joint_thumb_dynamixel_crank": 0.0,
            "Joint_thumb_crank_pusher": 0.0,
            
            "Joint_thumb_rotation":0.0,
            # New Robot
            "Joint_thumb_abduction":0.0,
            # "Joint_thumb_abduction":1.569,
            
            "Joint_thumb_flexion":0.0,
            "Joint_thumb_finray_proxy":0.0,

            "Joint_right_dynamixel_crank": 0.0,
            "Joint_right_crank_pusher": 0.0,
            
            "Joint_right_abduction":0.0,
            
            "Joint_right_flexion":0.0,
            "Joint_right_finray_proxy":0.0,
            
            "Joint_left_dynamixel_crank": 0.0, 
            "Joint_left_crank_pusher": 0.0, 
            
            "Joint_left_abduction":0.0,
            
            "Joint_left_flexion":0.0,
            "Joint_left_finray_proxy":0.0,
        # },
        # joint_pos={
            # Open Kinematick
            # "lbr_iiwa_joint_1": -0.036,
            # "lbr_iiwa_joint_2": 0.6,
            # "lbr_iiwa_joint_3": 2.966,
            # "lbr_iiwa_joint_4": 1.906,
            # "lbr_iiwa_joint_5": 2.966,
            # "lbr_iiwa_joint_6": -1,
            # "lbr_iiwa_joint_7": -2.926,
            # # "Joint_.*": 0.0,
            # "Joint_thumb_rotation":1.57,
            # "Joint_thumb_abduction":0.0,
            # "Joint_thumb_flexion":0.0,
            # "Joint_thumb_finray_proxy":0.0,
            # "Joint_right_abduction":0.0,
            # "Joint_right_flexion":0.0,
            # "Joint_right_finray_proxy":0.0,
            # "Joint_left_abduction":0.0,
            # "Joint_left_flexion":0.0,
            # "Joint_left_finray_proxy":0.0,
        
        # Grasp from top            
            # "lbr_iiwa_joint_1": -0.036,
            # "lbr_iiwa_joint_2": 0.6,
            # "lbr_iiwa_joint_3": 2.966,
            # "lbr_iiwa_joint_4": 1.0,
            # "lbr_iiwa_joint_5": -2.966,
            # "lbr_iiwa_joint_6": 1.57,
            # "lbr_iiwa_joint_7": -2.926,
            # "Joint_thumb_dynamixel_crank": 0.0,
            # "Joint_thumb_crank_pusher": 0.0,
            
            # "Joint_thumb_rotation":0.0,
            # "Joint_thumb_abduction":1.569,
            
            # "Joint_thumb_flexion":0.0,
            # "Joint_thumb_finray_proxy":0.0,

            # "Joint_right_dynamixel_crank": 0.0,
            # "Joint_right_crank_pusher": 0.0,
            
            # "Joint_right_abduction":0.0,
            
            # "Joint_right_flexion":0.0,
            # "Joint_right_finray_proxy":0.0,
            
            # "Joint_left_dynamixel_crank": 0.0, 
            # "Joint_left_crank_pusher": 0.0, 
            
            # "Joint_left_abduction":0.0,
            
            # "Joint_left_flexion":0.0,
            # "Joint_left_finray_proxy":0.0,

        },
    ),
    actuators={
        "kuka_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["lbr_iiwa_joint_[1-2]"],
            effort_limit_sim=320.0,
            velocity_limit_sim=85 * np.pi / 180.0,
            stiffness=300.0,
            damping=15.0,
        ),
        "kuka_forearm_1": ImplicitActuatorCfg(
            joint_names_expr=["lbr_iiwa_joint_3"],
            effort_limit_sim=176.0,
            velocity_limit_sim= 100 * np.pi / 180.0,
            stiffness=170.0,
            damping=8.0,
        ),
        "kuka_forearm_2": ImplicitActuatorCfg(
            joint_names_expr=["lbr_iiwa_joint_4"],
            effort_limit_sim=176.0,
            velocity_limit_sim= 75 * np.pi / 180.0,
            stiffness=170.0,
            damping=8.0,
        ),
        "kuka_forearm_3": ImplicitActuatorCfg(
            joint_names_expr=["lbr_iiwa_joint_5"],
            effort_limit_sim=110.0,
            velocity_limit_sim= 130 * np.pi / 180.0,
            stiffness=130.0,
            damping=6.0,
        ),
        "kuka_wrist": ImplicitActuatorCfg(
            joint_names_expr=["lbr_iiwa_joint_[6-7]"],
            effort_limit_sim=40.0,
            velocity_limit_sim= 135 * np.pi / 180.0,
            stiffness=20.0,
            damping=2.0,
        ),
        # "cringe_hand": ImplicitActuatorCfg(
        #     joint_names_expr=["Joint_.*_abduction", "Joint_.*_dynamixel_crank", "Joint_.*_rotation"],
        #     effort_limit_sim=1.5,
        #     velocity_limit_sim=6.17,
        #     stiffness=2e3,
        #     damping=1e2,
        # ),
        # "cringe_hand": ImplicitActuatorCfg(
        #     joint_names_expr=["Joint_.*"],
        #     effort_limit_sim=1.5*3,
        #     velocity_limit_sim=6.17,
        #     stiffness=2e3,
        #     damping=1e2,
        # ),
        "cringe_hand_active": ImplicitActuatorCfg(
            joint_names_expr=["Joint_.*_abduction", "Joint_.*_dynamixel_crank", "Joint_.*_rotation"],
            effort_limit_sim=1.5,
            velocity_limit_sim=6.17,
            stiffness=2e3,
            damping=1e2,
        ),
        "cringe_hand_spring": ImplicitActuatorCfg(
            joint_names_expr=["Joint_.*_finray_proxy"],
            effort_limit_sim=0.001,
            velocity_limit_sim=20,
            stiffness=0.15,
            damping=0,
        ),
        "cringe_hand_passive": ImplicitActuatorCfg(
            joint_names_expr=["Joint_.*_crank_pusher", "Joint_.*_flexion"],
            effort_limit_sim=50,
            velocity_limit_sim=20,
            stiffness=0,
            damping=0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


IIWA_CRINGE_CFG_HIGH_PD_CFG = IIWA_CRINGE_CFG.copy()
IIWA_CRINGE_CFG_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
IIWA_CRINGE_CFG_HIGH_PD_CFG.actuators["kuka_shoulder"].stiffness = 400.0
IIWA_CRINGE_CFG_HIGH_PD_CFG.actuators["kuka_shoulder"].damping = 80.0
IIWA_CRINGE_CFG_HIGH_PD_CFG.actuators["kuka_forearm_1"].stiffness = 400.0
IIWA_CRINGE_CFG_HIGH_PD_CFG.actuators["kuka_forearm_1"].damping = 80.0
IIWA_CRINGE_CFG_HIGH_PD_CFG.actuators["kuka_forearm_2"].stiffness = 400.0
IIWA_CRINGE_CFG_HIGH_PD_CFG.actuators["kuka_forearm_2"].damping = 80.0
IIWA_CRINGE_CFG_HIGH_PD_CFG.actuators["kuka_forearm_3"].stiffness = 400.0
IIWA_CRINGE_CFG_HIGH_PD_CFG.actuators["kuka_forearm_3"].damping = 80.0
IIWA_CRINGE_CFG_HIGH_PD_CFG.actuators["kuka_wrist"].stiffness = 400.0
IIWA_CRINGE_CFG_HIGH_PD_CFG.actuators["kuka_wrist"].damping = 80.0

"""Configuration of Kuka iiwa r14 with egor cringe hand robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
