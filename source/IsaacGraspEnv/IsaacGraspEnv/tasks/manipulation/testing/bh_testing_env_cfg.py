from dataclasses import MISSING
from isaaclab.assets import AssetBaseCfg, RigidObjectCollectionCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, TiledCameraCfg
import isaaclab.sim as sim_utils
from IsaacGraspEnv.robots.hand_barret.barret_hand_cfg import BH_CFG
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.utils import configclass

import torch
from scipy.spatial.transform import Rotation as R
import gymnasium as gym
import numpy as np


CAM_POS = torch.tensor([[3.5, 0.0, 3.5]])
CAM_ROT = R.from_euler("zyx", [90, -130, 0], degrees=True).as_quat()[[3, 0, 1, 2]]


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
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    contact_sensor_2 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Hand/bh_finger_23_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    contact_sensor_3 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Hand/bh_finger_33_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )


class BHTestingCamSceneCfg(BHTestingSceneCfg):
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        height=480,
        width=720,
        data_types=[
            "rgb",
            "distance_to_camera",
            "instance_segmentation_fast",
        ],
        colorize_instance_segmentation=False,
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
class BHTestingEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 5.0
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(14,))
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(14 + 7,))

    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    scene: InteractiveSceneCfg = BHTestingSceneCfg(num_envs=1, env_spacing=2)

    reward_coefs = {
        "reach": 1.0,
        "contact": 0.0,
        "lift": 0.0,
    }
