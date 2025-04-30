from __future__ import annotations

import math
import random
import torch
from typing import TYPE_CHECKING, Literal, List
import pathlib


import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import (
    EventTermCfg,
    ManagerTermBase,
    SceneEntityCfg,
    ObservationTermCfg,
)
from isaaclab.utils.math import sample_uniform

from isaaclab.utils.math import (
    subtract_frame_transforms,
)
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# def sample_object_poses(
#     num_objects: int,
#     min_separation: float = 0.0,
#     pose_range: dict[str, tuple[float, float]] = {},
#     max_sample_tries: int = 5000,
# ):
#     range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
#     pose_list = []

#     for i in range(num_objects):
#         for j in range(max_sample_tries):
#             sample = [random.uniform(range[0], range[1]) for range in range_list]

#             # Accept pose if it is the first one, or if reached max num tries
#             if len(pose_list) == 0 or j == max_sample_tries - 1:
#                 pose_list.append(sample)
#                 break

#             # Check if pose of object is sufficiently far away from all other objects
#             separation_check = [math.dist(sample[:3], pose[:3]) > min_separation for pose in pose_list]
#             if False not in separation_check:
#                 pose_list.append(sample)
#                 break

#     return pose_list


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

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        self.asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("object"))
        self.asset = env.scene[self.asset_cfg.name]

        self.t_assets_indices = torch.arange(
            self.asset.num_objects * self.asset.num_instances
        )

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

        self.asset_rest_delta_xy = torch.stack([x.flatten(), y.flatten()]).to(device=env.device).T
        
        self.gravity_vec_w = self.asset.data.GRAVITY_VEC_W.clone()

        self.asset_default_mass = self.asset.data.default_mass.clone().to(device=env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        out_focus_state: torch.Tensor,
        pose_range: dict[str, tuple[float, float]] = {},
    ):

        # List of rigid objects in focus for each env (dim = [num_envs, num_rigid_objects])

        # flags_activating_sim = self.mem_flag_disactive_sim_objs.clone()
        # flags_disabling_sim = self.mem_flag_disactive_sim_objs.clone()

        t_flag_disable_sim = self.asset.root_physx_view.get_disable_simulations()

        t_flag_disable_sim = self.asset.reshape_view_to_data(t_flag_disable_sim)

        flags_activating_sim = t_flag_disable_sim.clone()
        flags_disabling_sim = t_flag_disable_sim.clone()

        for env_id in env_ids.tolist():
            flags_activating_sim[env_id, :] = torch.zeros(self.asset.num_objects, 1)
            flags_disabling_sim[env_id, :] = torch.ones(self.asset.num_objects, 1)

        # self.asset.root_physx_view.set_disable_simulations(self.asset.reshape_data_to_view(flags_activating_sim), self.t_assets_indices)
        # TODO: Need to check influence on simulation. The extra simulation step can add upredicatable behavoir
        # env.sim.step()

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

            object_states[:, :2] = self.asset_rest_delta_xy[: self.asset.num_objects, :] + env.scene.env_origins[cur_env, 0:2]
            gravity_vec_w = self.gravity_vec_w[cur_env].clone()
            # gravity_vec_b, __ = subtract_frame_transforms(
            #     torch.zeros(self.asset.num_objects, 3, device=env.device), self.asset.data.object_quat_w[cur_env], self.gravity_vec_w[cur_env]
            #     )

            pose_tensor = torch.tensor([pose_list[0]], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(
                pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5]
            )
            object_states[object_id, 0:3] = positions
            object_states[object_id, 3:7] = orientations

            gravity_vec_w[object_id, :] = torch.zeros(
                3, device=env.device
            )

            self.asset.write_object_state_to_sim(
                object_state=object_states,
                env_ids=torch.tensor([cur_env], device=env.device),
            )
            self.list_id_rigid_objects_in_focus[cur_env][0] = object_id

            self.asset.set_external_force_and_torque(
                torch.mul(
                    gravity_vec_w, self.asset_default_mass[cur_env] * env.cfg.sim.gravity[2]
                ),
                torch.zeros(self.asset.num_objects, 3, device=env.device),
                env_ids=torch.tensor([cur_env], device=env.device),
            )

            # env.scene.write_data_to_sim()
            # env.sim.forward()
            flags_disabling_sim[cur_env, object_id] = 0
            # TODO: Need to check influence on simulation. The extra simulation step can add upredicatable behavoir
            # env.sim.step()

        # env.sim.step()
        # self.asset.root_physx_view.set_disable_simulations(
        #     self.asset.reshape_data_to_view(flags_disabling_sim), self.t_assets_indices
        # )

        self.mem_flag_disactive_sim_objs = flags_disabling_sim.clone()
        # # TODO: Need to check influence on simulation. The extra simulation step can add upredicatable behavoir


# def randomize_rigid_objects_in_focus(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor,
#     asset_cfgs: list[SceneEntityCfg],
#     out_focus_state: torch.Tensor,
#     min_separation: float = 0.0,
#     pose_range: dict[str, tuple[float, float]] = {},
#     max_sample_tries: int = 5000,
# ):
#     if env_ids is None:
#         return

#     # List of rigid objects in focus for each env (dim = [num_envs, num_rigid_objects])
#     t_indces_asssets = []
#     flags_disabling_sim_assets = []

#     for asset_idx in range(len(asset_cfgs)):
#         asset_cfg = asset_cfgs[asset_idx]
#         asset = env.scene[asset_cfg.name]
#         flags_activating_sim = torch.full((asset.num_objects, asset.num_instances, 1), fill_value=0, dtype=torch.uint8)#.to(env.device)


#         t_indices_obj = torch.arange(asset.num_objects * asset.num_instances)#.to(env.device)
#         asset.root_physx_view.set_disable_simulations(flags_activating_sim.flatten(), t_indices_obj)
#         t_indces_asssets.append(t_indices_obj)
#         flags_disabling_sim_assets.append((flags_activating_sim.clone()+1))
#     env.sim.step()
#     env.rigid_objects_in_focus = []

#     for cur_env in env_ids.tolist():
#         # Sample in focus object poses
#         pose_list = sample_object_poses(
#             num_objects=len(asset_cfgs),
#             min_separation=min_separation,
#             pose_range=pose_range,
#             max_sample_tries=max_sample_tries,
#         )


#         selected_ids = []
#         for asset_idx in range(len(asset_cfgs)):
#             asset_cfg = asset_cfgs[asset_idx]
#             asset = env.scene[asset_cfg.name]


#             # Randomly select an object to bring into focus
#             object_id = random.randint(0, asset.num_objects - 1)
#             selected_ids.append(object_id)
#             # Create object state tensor
#             object_states = torch.stack([out_focus_state] * asset.num_objects).to(device=env.device)
#             pose_tensor = torch.tensor([pose_list[asset_idx]], device=env.device)
#             positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
#             orientations = math_utils.quat_from_euler_xyz(pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5])
#             object_states[object_id, 0:3] = positions
#             object_states[object_id, 3:7] = orientations

#             asset.write_object_state_to_sim(
#                 object_state=object_states, env_ids=torch.tensor([cur_env], device=env.device)
#             )
#             # asset.root_physx_view.set_coms(torch.stack([torch.Tensor([10.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])] * asset.num_objects).to(device=env.device), t_indices_obj)
#             env.rigid_objects_in_focus.append(selected_ids)

#             # asset.reset()

#             flags_disabling_sim_assets[asset_idx][object_id, cur_env] = 0
#     # TODO: Need to check influence on simulation. The extra simulation step can add upredicatable behavoir
#     env.sim.step()
#     for asset_idx in range(len(asset_cfgs)):
#         asset_cfg = asset_cfgs[asset_idx]
#         asset = env.scene[asset_cfg.name]
#         asset.root_physx_view.set_disable_simulations(flags_disabling_sim_assets[asset_idx].flatten(), t_indces_asssets[asset_idx])

#     # # TODO: Need to check influence on simulation. The extra simulation step can add upredicatable behavoir
#     # env.sim.step()
