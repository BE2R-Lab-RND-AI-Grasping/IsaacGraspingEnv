from __future__ import annotations

import math
import random
import torch
from typing import TYPE_CHECKING, Literal, List
import pathlib


import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg, ObservationTermCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv



def sample_object_poses(
    num_objects: int,
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    pose_list = []

    for i in range(num_objects):
        for j in range(max_sample_tries):
            sample = [random.uniform(range[0], range[1]) for range in range_list]

            # Accept pose if it is the first one, or if reached max num tries
            if len(pose_list) == 0 or j == max_sample_tries - 1:
                pose_list.append(sample)
                break

            # Check if pose of object is sufficiently far away from all other objects
            separation_check = [math.dist(sample[:3], pose[:3]) > min_separation for pose in pose_list]
            if False not in separation_check:
                pose_list.append(sample)
                break

    return pose_list

def randomize_rigid_objects_in_focus(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    out_focus_state: torch.Tensor,
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    if env_ids is None:
        return

    # List of rigid objects in focus for each env (dim = [num_envs, num_rigid_objects])
    env.rigid_objects_in_focus = []

    for cur_env in env_ids.tolist():
        # Sample in focus object poses
        pose_list = sample_object_poses(
            num_objects=len(asset_cfgs),
            min_separation=min_separation,
            pose_range=pose_range,
            max_sample_tries=max_sample_tries,
        )


        selected_ids = []
        for asset_idx in range(len(asset_cfgs)):
            asset_cfg = asset_cfgs[asset_idx]
            asset = env.scene[asset_cfg.name]

            flgas_activating_sim = torch.full((asset.num_objects, 1), fill_value=0, dtype=torch.uint8).to(env.device)
            t_indices_obj = torch.arange(asset.num_objects)
            asset.root_physx_view.set_disable_simulations(flgas_activating_sim, t_indices_obj)
            

            # Randomly select an object to bring into focus
            object_id = random.randint(0, asset.num_objects - 1)
            selected_ids.append(object_id)
            # Create object state tensor
            object_states = torch.stack([out_focus_state] * asset.num_objects).to(device=env.device)
            pose_tensor = torch.tensor([pose_list[asset_idx]], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5])
            object_states[object_id, 0:3] = positions
            object_states[object_id, 3:7] = orientations

            asset.write_object_state_to_sim(
                object_state=object_states, env_ids=torch.tensor([cur_env], device=env.device)
            )
            # asset.root_physx_view.set_coms(torch.stack([torch.Tensor([10.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])] * asset.num_objects).to(device=env.device), t_indices_obj)
            asset.reset()
            # TODO: Need to check influence on simulation. The extra simulation step can add upredicatable behavoir
            env.sim.step()
            
            flgas_disabling_sim = torch.full((asset.num_objects, 1), fill_value=1, dtype=torch.uint8).to(env.device)
            flgas_disabling_sim[object_id] = 0
            
            asset.root_physx_view.set_disable_simulations(flgas_disabling_sim, t_indices_obj)

        env.rigid_objects_in_focus.append(selected_ids)