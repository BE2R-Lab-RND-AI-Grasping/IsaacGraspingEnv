

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
        '''Class to compute the displacement of the ojbect relative to its previous position
        Args:
            cfg (RewardTermCfg): Configuration for the reward term.
            env (ManagerBasedRLEnv): The environment instance.
        '''

        super().__init__(cfg, env)

        self.object_cfg = cfg.params["object_cfg"]

        self.prev_object_pos = torch.full((env.num_envs, 3), 0).to(env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        """Penalize the displacement of the object from its previous position.
        
        Args:
            env (ManagerBasedRLEnv): The environment instance.
            object_cfg (SceneEntityCfg, optional): Configuration of the object. Defaults to SceneEntityCfg("object").
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
        """Class to compute the norm of vectors in the observation space.

        Args:
            cfg (RewardTermCfg): Configuration for the reward term.
            env (ManagerBasedRLEnv): The environment instance.
        """
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
        
        """Compute the norm of vectors in the observation space.

        Args:
            env (ManagerBasedRLEnv): The environment instance.
            name_obs_vector (str): The name of the observation term is defined in the configuration.

        Returns:
            torch.Tensor: The computed norm of the observation vectors.
        """

        vectors = self.unpack_vectors4obs(env).reshape(env.num_envs,-1,3)
        
        norm_vec = torch.linalg.norm(vectors, dim=-1).sum(dim=1)
        
        return norm_vec
    
    