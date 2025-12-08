# Copyright (c) 2025-2027, BE2R Lab
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Kuka iiwa r14 with shadow DEX-EE gripper"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##
from pathlib import Path
import numpy as np


relative_path = (
    "../../source/Iiwa14_DEXEE_Grasp/Iiwa14_DEXEE_Grasp/Robots/iiwa_DEXEE2.usd"
)
ABSOLUTE_PATH = Path(relative_path).resolve()

# INIT_Q_IIWA = np.array([ 0.3228,  1.0685, -0.3784, -1.0638, -0.1891,  1.1131,  0.2125])
INIT_Q_IIWA = np.array([0.3228, 0.0685, -0.3784, -1.0638, -0.1891, 1.1131, 0.2125])

INIT_Q_IIWA = INIT_Q_IIWA.tolist()

RELATIVE_PATH2 = "../../source/Iiwa14_DEXEE_Grasp/Data/core-bowl-a593e8863200fdb0664b3b9b23ddfcbc/coacd/core-bowl-a593e8863200fdb0664b3b9b23ddfcbc.npy"
# RELATIVE_PATH2 = "../../source/Iiwa14_DEXEE_Grasp/Data/ddg-gd_drill_poisson_000/coacd/ddg-gd_drill_poisson_000.npy"

ABSOLUTE_PATH2 = Path(RELATIVE_PATH2).resolve().as_posix()

data = np.load(ABSOLUTE_PATH2, allow_pickle=True)
grasp = data[0]
final_pose = grasp["qpos"]
print(final_pose)
IIWA14_DEXEE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ABSOLUTE_PATH.as_posix(),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=10.0,
            max_angular_velocity=36.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=0,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0015),
        # semantic_tags = [("class","robot"), ("color", "orange")],
    ),
    # -0.036,  1.204,   2.9670658, 1.906,    2.9671504,  -1.555,-2.926
    init_state=ArticulationCfg.InitialStateCfg(
        # pos=[0.0, 0.0, 0.0], rot=[1,0,0,0],
        # Grasp from front
        joint_pos={
            "joint1": INIT_Q_IIWA[0],
            "joint2": INIT_Q_IIWA[1],
            "joint3": INIT_Q_IIWA[2],
            "joint4": INIT_Q_IIWA[3],
            "joint5": INIT_Q_IIWA[4],
            "joint6": INIT_Q_IIWA[5],
            "joint7": INIT_Q_IIWA[6],
            "F0_J0": final_pose["F0_J0"],
            "F0_J1": final_pose["F0_J1"],
            "F0_J2": final_pose["F0_J2"],
            "F0_J3": final_pose["F0_J3"],
            "F1_J0": final_pose["F1_J0"],
            "F1_J1": final_pose["F1_J1"],
            "F1_J2": final_pose["F1_J2"],
            "F1_J3": final_pose["F1_J3"],
            "F2_J0": final_pose["F2_J0"],
            "F2_J1": final_pose["F2_J1"],
            "F2_J2": final_pose["F2_J2"],
            "F2_J3": final_pose["F2_J3"],
            # "F0_J0":0.199,
            # "F0_J1":-1.394,
            # "F0_J2":0,
            # "F0_J3":0.0,
            # "F1_J0":-0.019,
            # "F1_J1":-1.394,
            # "F1_J2":0.0,
            # "F1_J3":0.0,
            # "F2_J0":0.105929,
            # "F2_J1":-1.394,
            # "F2_J2":0.0,
            # "F2_J3":0.0,
        },
    ),
    actuators={
        "kuka_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-2]"],
            effort_limit_sim=300.0,
            # velocity_limit_sim=85 * np.pi / 180.0,
            stiffness=300.0,
            damping=45.0,
            friction=1.0,
        ),
        "kuka_forearm_1": ImplicitActuatorCfg(
            joint_names_expr=["joint3"],
            effort_limit_sim=300.0,
            # velocity_limit_sim= 100 * np.pi / 180.0,
            stiffness=300.0,
            damping=45.0,
            friction=1.0,
        ),
        "kuka_forearm_2": ImplicitActuatorCfg(
            joint_names_expr=["joint4"],
            effort_limit_sim=300.0,
            # velocity_limit_sim= 75 * np.pi / 180.0,
            stiffness=300.0,
            damping=45.0,
            friction=1.0,
        ),
        "kuka_forearm_3": ImplicitActuatorCfg(
            joint_names_expr=["joint5"],
            effort_limit_sim=300.0,
            # velocity_limit_sim= 130 * np.pi / 180.0,
            stiffness=100.0,
            damping=20.0,
            friction=1.0,
        ),
        "kuka_wrist_1": ImplicitActuatorCfg(
            joint_names_expr=["joint6"],
            effort_limit_sim=300.0,
            # velocity_limit_sim= 135 * np.pi / 180.0,
            stiffness=50.0,
            damping=15.0,
            friction=1.0,
        ),
        "kuka_wrist_2": ImplicitActuatorCfg(
            joint_names_expr=["joint7"],
            effort_limit_sim=300.0,
            # velocity_limit_sim= 135 * np.pi / 180.0,
            stiffness=25.0,
            damping=15.0,
            friction=1.0,
        ),
        "F0_J0": ImplicitActuatorCfg(
            joint_names_expr=["F0_J0"],
            effort_limit_sim=2.5,
            velocity_limit_sim=1.5,
            stiffness=60,
            damping=20,
        ),
        "F0_J1": ImplicitActuatorCfg(
            joint_names_expr=["F0_J1"],
            effort_limit_sim=2.5,
            velocity_limit_sim=1.5,
            stiffness=60,
            damping=20,
        ),
        "F0_J2": ImplicitActuatorCfg(
            joint_names_expr=["F0_J2"],
            effort_limit_sim=2.5,
            velocity_limit_sim=1.5,
            stiffness=60,
            damping=20,
        ),
        "F0_J3": ImplicitActuatorCfg(
            joint_names_expr=["F0_J3"],
            effort_limit_sim=2.5,
            velocity_limit_sim=1.5,
            stiffness=60,
            damping=20,
        ),
        "F1_J0": ImplicitActuatorCfg(
            joint_names_expr=["F1_J0"],
            effort_limit_sim=1.5 * 3,
            velocity_limit_sim=6.17,
            stiffness=2.8e3,
            damping=0.03e3,
        ),
        "F1_J1": ImplicitActuatorCfg(
            joint_names_expr=["F1_J1"],
            effort_limit_sim=1.5 * 3,
            velocity_limit_sim=6.17,
            stiffness=2.5e3,
            damping=0.02e3,
        ),
        "F1_J2": ImplicitActuatorCfg(
            joint_names_expr=["F1_J2"],
            effort_limit_sim=1.5 * 3,
            velocity_limit_sim=6.17,
            stiffness=1.1e3,
            damping=0.01e3,
        ),
        "F1_J3": ImplicitActuatorCfg(
            joint_names_expr=["F1_J3"],
            effort_limit_sim=1.5 * 3,
            velocity_limit_sim=6.17,
            stiffness=0.6e3,
            damping=0.008e3,
        ),
        "F2_J0": ImplicitActuatorCfg(
            joint_names_expr=["F2_J0"],
            effort_limit_sim=1.5 * 3,
            velocity_limit_sim=6.17,
            stiffness=2.8e3,
            damping=0.03e3,
        ),
        "F2_J1": ImplicitActuatorCfg(
            joint_names_expr=["F2_J1"],
            effort_limit_sim=1.5 * 3,
            velocity_limit_sim=6.17,
            stiffness=2.5e3,
            damping=0.02e3,
        ),
        "F2_J2": ImplicitActuatorCfg(
            joint_names_expr=["F2_J2"],
            effort_limit_sim=1.5 * 3,
            velocity_limit_sim=6.17,
            stiffness=1.1e3,
            damping=0.01e3,
        ),
        "F2_J3": ImplicitActuatorCfg(
            joint_names_expr=["F2_J3"],
            effort_limit_sim=1.5 * 3,
            velocity_limit_sim=6.17,
            stiffness=0.6e3,
            damping=0.008e3,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
