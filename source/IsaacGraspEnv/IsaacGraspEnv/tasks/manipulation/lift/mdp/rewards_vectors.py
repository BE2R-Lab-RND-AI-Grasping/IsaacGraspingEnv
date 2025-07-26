

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg, RewardTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .observations import (
    instance_randomize_obj_positions_in_robot_world_frame as get_obj_pos_w,

)

from .observations_vectors import instance_vectors_joint_hand_key_points

class instance_object_displacement(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):

        super().__init__(cfg, env)

        self.object_cfg = cfg.params["object_cfg"]

        self.prev_object_pos = torch.full((env.num_envs, 3), 0).to(env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Penalize joint velocities on the articulation using L2 squared kernel.

        NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
        """

        curr_obj_pos = get_obj_pos_w(env, self.object_cfg)
        # extract the used quantities (to enable type-hinting)
        reward = torch.sum(
            torch.abs(curr_obj_pos - self.prev_object_pos),
            dim=1,
        )

        self.prev_object_pos = curr_obj_pos.clone()

        return reward

class instance_vectors_norm(ManagerTermBase):
    
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):

        super().__init__(cfg, env)

        self.name_obs_vector = cfg.params["name_obs_vector"]
        self.prev_object_pos = torch.full((env.num_envs, 3), 0).to(env.device)
        
        # self.obs_vectors = instance_vectors_joint_hand_key_points()
        
        id_obs_vector = env.observation_manager.active_terms["policy"].index(self.name_obs_vector)
        size_obs = env.observation_manager.group_obs_term_dim["policy"][id_obs_vector][0]
        index_obs_vector = sum([env.observation_manager.group_obs_term_dim["policy"][i][0] for i in range(id_obs_vector)])
        
        
        self.unpack_vectors4obs = lambda env: env.observation_manager._obs_buffer["policy"][:,index_obs_vector:index_obs_vector+size_obs]


    def __call__(
        self,
        env: ManagerBasedRLEnv,
        name_obs_vector: str, 
    ) -> torch.Tensor:
        
        
        vectors = self.unpack_vectors4obs(env).reshape(env.num_envs,-1,3)
        
        norm_vec = torch.linalg.norm(vectors, dim=-1).sum(dim=1)
        
        return norm_vec
    
    