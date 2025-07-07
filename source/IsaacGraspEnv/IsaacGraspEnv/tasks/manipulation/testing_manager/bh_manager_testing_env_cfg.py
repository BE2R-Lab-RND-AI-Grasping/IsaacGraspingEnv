from math import pi
from scipy.spatial.transform import Rotation
import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from dataclasses import MISSING
from isaaclab.assets import AssetBaseCfg, RigidObjectCollectionCfg
from isaaclab.envs.mdp import EventTermCfg, RewardTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import (
    CameraCfg,
    ContactSensorCfg,
    FrameTransformerCfg,
    TiledCameraCfg,
)
import isaaclab.sim as sim_utils
from IsaacGraspEnv.robots.hand_barret.barret_hand_cfg import BH_CFG
from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.managers import (
    CommandManager,
    ObservationGroupCfg,
    ObservationTermCfg as ObsTerm,
    TerminationTermCfg,
)
from IsaacGraspEnv.tasks.manipulation.lift import mdp


class BHTestingSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot = BH_CFG.replace(prim_path="{ENV_REGEX_NS}/Hand")
    object = RigidObjectCollectionCfg(rigid_objects=MISSING)
    contact_sensor_1 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Hand/bh_finger_13_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_*"],
    )
    contact_sensor_2 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Hand/bh_finger_23_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_*"],
    )
    contact_sensor_3 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Hand/bh_finger_33_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_*"],
    )
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Hand/root_link_0",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Hand/base_link",
                name="end_effector",
            ),
        ],
    )
    thumb_ft_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Hand/root_link_0",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Hand/bh_finger_33_link",
                name="fingertips_thumb",
            ),
        ],
    )

    right_ft_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Hand/root_link_0",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Hand/bh_finger_23_link",
                name="fingertips_right",
            ),
        ],
    )

    left_ft_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Hand/root_link_0",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Hand/bh_finger_13_link",
                name="fingertips_left",
            ),
        ],
    )


CAM_POS = torch.tensor([[3.5, 0.0, 3.5]])
CAM_ROT = Rotation.from_euler("zyx", [90, -130, 0], degrees=True).as_quat()[
    [3, 0, 1, 2]
]


class BHTestingCamSceneCfg(BHTestingSceneCfg):
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        height=480,
        width=720,
        colorize_semantic_segmentation=False,
        semantic_filter="class: object | robot",
        data_types=[
            "rgb",
            "distance_to_image_plane",
            "semantic_segmentation",
            "distance_to_camera",
        ],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=tuple(CAM_POS.tolist()),
            rot=tuple(CAM_ROT.tolist()),
            convention="ros",
        ),
    )


@configclass
class ActionsCfg:
    cartesian_action = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=["base_joint.*"], preserve_order=True
    )
    gripper_action = mdp.JointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=["bh.*"],
        rescale_to_limits=True,
    )


@configclass
class ProPrioceptionObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""

        ee_frame = ObsTerm(func=mdp.frame_in_robot_root_frame)
        fingertips_positions = ObsTerm(func=mdp.pos_fingertips_root_frame)
        joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # not needed if object is fixed in space
        # object_position = ObsTerm(
        #     func=mdp.instance_randomize_obj_positions_in_robot_root_frame
        # )
        # object_quat = ObsTerm(
        #     func=mdp.instance_randomize_obj_orientations_in_robot_root_frame
        # )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RGBDObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""

        image = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "rgb"},
        )
        depth = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "data_type": "distance_to_camera",
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class PointCloudObservationsCfg:
    @configclass
    class PointCloudPolicyCfg(ObservationGroupCfg):
        point_cloud = ObsTerm(
            func=mdp.depth2point_cloud,
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "data_type": "distance_to_camera",
            },
        )

    # observation groups
    policy: PointCloudPolicyCfg = PointCloudPolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    randomize_cubes_in_focus = EventTermCfg(
        func=mdp.randomize_rigid_object_in_focus,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "out_focus_state": torch.tensor(
                [0.0, 0.0, 10.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ),
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.3, 0.8),
                "roll": (pi / 4, 3 * pi / 4),
                "pitch": (-pi / 4, pi / 4),
                "yaw": (pi / 4, 3 * pi / 4),
            },
            "only_pose": True,
        },
    )

    reset_all = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    is_alive = RewardTermCfg(func=mdp.is_alive, weight=1.0)


@configclass
class BHTestingManagerEnvCfg(ManagerBasedRLEnvCfg):
    scene = BHTestingSceneCfg(num_envs=1, env_spacing=10)
    actions = ActionsCfg()
    observations = ProPrioceptionObservationsCfg()
    # commands = CommandsCfg()
    # MDP settings
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 3.0
        # simulation settings
        self.sim.dt = 0.01  # 1.0 /120.0  # 100Hz
        self.sim.render_interval = self.decimation


@configclass
class BHTestingRGBDManagerEnvCfg(BHTestingManagerEnvCfg):
    scene = BHTestingCamSceneCfg(num_envs=1, env_spacing=10)
    observations = RGBDObservationsCfg()


@configclass
class BHTestingPointCloudManagerEnvCfg(BHTestingManagerEnvCfg):
    scene = BHTestingCamSceneCfg(num_envs=1, env_spacing=10)
    observations = PointCloudObservationsCfg()
