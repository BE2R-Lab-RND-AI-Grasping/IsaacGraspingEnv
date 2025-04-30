from __future__ import annotations

import math
import random
import torch
from typing import TYPE_CHECKING, Literal, List
import pathlib


import isaaclab.utils.math as math_utils
from isaaclab.managers import (
    EventTermCfg,
    ManagerTermBase,
    SceneEntityCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def sample_object_poses(
    num_objects: int,
    pose_range: dict[str, tuple[float, float]] = {},
):
    range_list = [
        pose_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    pose_list = []

    for i in range(num_objects):
        sample = [random.uniform(range[0], range[1]) for range in range_list]
        pose_list.append(sample)

    return pose_list


class randomize_rigid_object_in_focus(ManagerTermBase):
    """Manager term for randomizing and activating a single rigid object 'in focus' per environment.

    This class is designed for simulated environments with multiple rigid objects.
    For each call, it selects a random rigid object in each specified environment, places (randomizes)
    it within a pose range, and ensures only that object is active in the physics simulation
    (the rest are placed out_focus_state with gravity compensation).

    Args:
        cfg: The configuration structure for this manager term, containing parameters needed for initialization.
        env: The simulation environment

    Attributes:
        asset_cfg: The asset configuration, specifying which scene entity RigidObjectCollection to work with.
        asset: The asset object as retrieved from the environment's scene.
        list_id_rigid_objects_in_focus: For each environment, a list of indices indicating the object currently in focus.
        asset_rest_delta_xy: Prepared 2D grid of offsets for positioning objects in the rest.
        gravity_vec_w: The world-frame gravity vectors for the object asset (cloned from simulation data).
        asset_default_mass: Default mass values for the asset, on the relevant device.
        mem_flag_disactive_sim_objs: Memory for flags to disable/enable simulation of objects (initialized in __call__).

    Example:
        # Initialize
        obj_manager = randomize_rigid_object_in_focus(cfg, env)
        # On event: randomize a batch of environments
        obj_manager(
            env,
            env_ids=torch.tensor([0, 1, 2]),
            asset_cfg=scene_cfg,
            out_focus_state=some_state_tensor,
            pose_range={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        )
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        self.asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("object"))
        self.asset = env.scene[self.asset_cfg.name]

        self.list_id_rigid_objects_in_focus = [[0] for __ in range(env.num_envs)]
        env.rigid_objects_in_focus = self.list_id_rigid_objects_in_focus

        # self.mem_flag_disactive_sim_objs = torch.full((self.asset.num_objects, self.asset.num_instances, 1), fill_value=0, dtype=torch.uint8)

        x = torch.linspace(
            -self.asset.num_objects / 2 * 0.15,
            self.asset.num_objects / 2 * 0.15,
            math.floor(self.asset.num_objects / 2),
        )
        y = torch.linspace(
            -self.asset.num_objects / 2 * 0.15,
            self.asset.num_objects / 2 * 0.15,
            math.floor(self.asset.num_objects / 2),
        )

        x, y = torch.meshgrid(y, x)

        self.asset_rest_delta_xy = (
            torch.stack([x.flatten(), y.flatten()]).to(device=env.device).T
        )

        self.gravity_vec_w = self.asset.data.GRAVITY_VEC_W.clone()

        self.asset_default_mass = self.asset.data.default_mass.clone().to(
            device=env.device
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        out_focus_state: torch.Tensor,
        pose_range: dict[str, tuple[float, float]] = {},
    ):

        for cur_env in env_ids.tolist():
            # Sample in focus object poses
            pose_list = sample_object_poses(
                num_objects=1,
                pose_range=pose_range,
            )

            # Randomly select an object to bring into focus
            object_id = random.randint(0, self.asset.num_objects - 1)
            # Create object state tensor
            object_states = torch.stack([out_focus_state] * self.asset.num_objects).to(
                device=env.device
            )
            # Place the rest object on grid in out focus state
            object_states[:, :2] = (
                self.asset_rest_delta_xy[: self.asset.num_objects, :]
                + env.scene.env_origins[cur_env, 0:2]
            )
            gravity_vec_w = self.gravity_vec_w[cur_env].clone()
            
            # gravity_vec_b, __ = subtract_frame_transforms(
            #     torch.zeros(self.asset.num_objects, 3, device=env.device), self.asset.data.object_quat_w[cur_env], self.gravity_vec_w[cur_env]
            #     )

            # Define place in focus
            pose_tensor = torch.tensor([pose_list[0]], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(
                pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5]
            )
            object_states[object_id, 0:3] = positions
            object_states[object_id, 3:7] = orientations

            gravity_vec_w[object_id, :] = torch.zeros(3, device=env.device)

            self.asset.write_object_state_to_sim(
                object_state=object_states,
                env_ids=torch.tensor([cur_env], device=env.device),
            )
            self.list_id_rigid_objects_in_focus[cur_env][0] = object_id

            # Gravity compensate
            self.asset.set_external_force_and_torque(
                torch.mul(
                    gravity_vec_w,
                    self.asset_default_mass[cur_env] * env.cfg.sim.gravity[2],
                ),
                torch.zeros(self.asset.num_objects, 3, device=env.device),
                env_ids=torch.tensor([cur_env], device=env.device),
            )