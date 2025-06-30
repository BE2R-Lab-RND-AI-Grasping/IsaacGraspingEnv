from pathlib import Path
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import (
    ArticulationRootPropertiesCfg,
    CollisionPropertiesCfg,
    RigidBodyPropertiesCfg,
    UrdfConverterCfg,
    UrdfFileCfg,
)

BH_URDF_REL_PATH = "source/IsaacGraspEnv/IsaacGraspEnv/robots/hand_barret/bh_alone.urdf"
BH_URDF_ABS_PATH = Path(BH_URDF_REL_PATH).resolve().as_posix()


BH_CFG = ArticulationCfg(
    spawn=UrdfFileCfg(
        asset_path=BH_URDF_ABS_PATH,
        fix_base=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.NaturalFrequencyGainsCfg(
                natural_frequency=1.0,
            )
        ),
        rigid_props=RigidBodyPropertiesCfg(
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
        articulation_props=ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=0,
        ),
        collision_props=CollisionPropertiesCfg(
            contact_offset=0.001, rest_offset=0.0015
        ),
        semantic_tags=[("class", "robot"), ("color", "orange")],
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0),
        joint_pos={
            "base_joint_z": 1.0,
            "base_joint_R": 3.14,
            "bh_j32_joint": (0 + 2.44) / 2,
            "bh_j33_joint": (0 + 0.84) / 2,
            "bh_j11_joint": (0 + 3.1416) / 2,
            "bh_j12_joint": (0 + 2.44) / 2,
            "bh_j13_joint": (0 + 0.84) / 2,
            "bh_j21_joint": (0 + 3.1416) / 2,
            "bh_j22_joint": (0 + 2.44) / 2,
            "bh_j23_joint": (0 + 0.84) / 2,
        },
    ),
    actuators={
        "base_act": ImplicitActuatorCfg(
            joint_names_expr=["base_joint.*"],
            stiffness=1000.0,
            damping=100.0,
        ),
        "finger_acts": ImplicitActuatorCfg(
            joint_names_expr=["bh.*"],
            stiffness=300.0,
            damping=15.0,
        ),
    },
)
