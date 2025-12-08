import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##
from pathlib import Path
import numpy as np

relative_path = "source/IsaacGraspEnv/IsaacGraspEnv/robots/hand_iiwa/hand_iiwa_free_open.usd"
ABSOLUTE_PATH = Path(relative_path).resolve()


DP_FLEX_FREE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ABSOLUTE_PATH.as_posix(),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
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
        pos = (0.0, 0.0, 0.1),
        # Grasp from front
        joint_pos={

            "Joint_thumb_rotation":0.0,
            # New Robot
            "Joint_thumb_abduction":0.0,
            
            "Joint_thumb_flexion":0.0,
            "Joint_thumb_finray_proxy":0.0,

            "Joint_right_abduction":0.0,
            
            "Joint_right_flexion":0.0,
            "Joint_right_finray_proxy":0.0,
            
            "Joint_left_abduction":0.0,
            
            "Joint_left_flexion":0.0,
            "Joint_left_finray_proxy":0.0,
        },
    ),
    actuators={
        "cringe_hand_active": ImplicitActuatorCfg(
            joint_names_expr=["Joint_.*"],
            effort_limit_sim=1.5*3,
            velocity_limit_sim=6.17,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)