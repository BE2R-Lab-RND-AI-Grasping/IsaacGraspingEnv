

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg, RewardTermCfg, ManagerTermBase
from isaaclab.utils.math import (
    quat_unique,
    subtract_frame_transforms,
    combine_frame_transforms,
    quat_mul,
    transform_points
)

import numpy as np

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .observations import (
    instance_randomize_obj_positions_in_robot_world_frame as get_obj_pos_w,
    instance_randomize_obj_orientations_in_world_frame as get_obj_quat_w,
)



from .observations_vectors import instance_vectors_joint_hand_key_points
    
def create_extractor_obs_term4vectors(env: ManagerBasedRLEnv, name_obs_vector: str):
    
    id_obs_vector = env.observation_manager.active_terms["policy"].index(name_obs_vector)
    size_obs = env.observation_manager.group_obs_term_dim["policy"][id_obs_vector][0]
    index_obs_vector = sum([env.observation_manager.group_obs_term_dim["policy"][i][0] for i in range(id_obs_vector)])
    
    unpack_vectors4obs = lambda env=env: env.observation_manager.compute_group("policy")[:,index_obs_vector:index_obs_vector+size_obs]
    
    return unpack_vectors4obs

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
        self.unpack_vectors4obs = create_extractor_obs_term4vectors(env, self.name_obs_vector)


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

        vectors = self.unpack_vectors4obs().reshape(env.num_envs,-1,3)
        
        norm_vec = torch.linalg.norm(vectors, dim=-1).sum(dim=1)
        
        return norm_vec
    

class distance_frame_orientation_to_target(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Class to compute the norm of quaternion in the observation space.

        Args:
            cfg (RewardTermCfg): Configuration for the reward term.
            env (ManagerBasedRLEnv): The environment instance.
        """
        super().__init__(cfg, env)

        self.observation_term = cfg.params["observation_term"]
        
        self.frame_cfg = cfg.params["frame_cfg"]
        self.frame: RigidObject = env.scene[self.frame_cfg.name]
        
        # self.obs_vectors = instance_vectors_joint_hand_key_points()
        self.unpack_vectors4obs = create_extractor_obs_term4vectors(env, self.observation_term)


    def __call__(
        self,
        env: ManagerBasedRLEnv,
        frame_cfg: SceneEntityCfg,
        observation_term: str = "relative_target_quat_current"
    ) -> torch.Tensor:

        target_quat_ee = self.unpack_vectors4obs()
        
        ee_quat_w = self.frame.data.target_quat_w.squeeze(1)
        
        w_quat_target = quat_mul(ee_quat_w,target_quat_ee) # R^w_ee @ R^ee_target
        
        res = 1 - torch.linalg.vecdot(ee_quat_w, w_quat_target)
        
        return res
    
class desired_contact_points_displacement(ManagerTermBase):
    
    
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Class to compute reward displacement contact points on object and amount of forces
        Reference: https://arxiv.org/pdf/2112.03028

        Args:
            cfg (RewardTermCfg): reward confing
            env (ManagerBasedRLEnv): environment
        """
        
        
        super().__init__(cfg, env)
        
        self.contact_key = cfg.params["contact_key"]
        self.grasping_reference_path = cfg.params["grasping_reference_path"]
        
        self.object_cfg = cfg.params["object_cfg"]
        self.object: RigidObjectCollection = env.scene[self.object_cfg.name]
        
        self.t_contact_terms_keys = tuple([key for key in cfg.params if key.find("cfgs") > -1])
        self.d_contact_sensor_cfgs = {key: cfg.params[key] for key in self.t_contact_terms_keys}
        self.d_contact_sensors = {key: env.scene[cfg.name] for key, cfg in self.d_contact_sensor_cfgs.items()}
        
        instance_contact_ref_pos_obj = self.mapping_reference_n_sim_hand(self.grasping_reference_path, self.t_contact_terms_keys).to(env.device)
        ref_contact_shape = instance_contact_ref_pos_obj.shape
        self.contact_ref_pos_obj = instance_contact_ref_pos_obj.squeeze(-2).repeat(env.num_envs, 1, 1)  # (num_envs, num_contact_sensors, 3)
        
    def mapping_reference_n_sim_hand(self, grasping_reference_path, ordered_keys):

        with open(grasping_reference_path, "rb") as f:

            grasping_reference = np.load(f, allow_pickle=True)


        ordered_contact_reference = []
        for grasp_ref in grasping_reference:
            one_contact_pos_ref = []
            for contact_sensor_key in ordered_keys:
                sensor_name = self.d_contact_sensor_cfgs[contact_sensor_key].name
                if sensor_name in grasp_ref[self.contact_key]:
                    one_contact_pos_ref.append(grasp_ref[self.contact_key][self.d_contact_sensor_cfgs[contact_sensor_key].name])
                else:
                    arr_nan = np.zeros((1,3))
                    arr_nan.fill(np.nan)
                    one_contact_pos_ref.append(arr_nan)

            ordered_contact_reference.append(one_contact_pos_ref)
            
        

        return torch.Tensor(ordered_contact_reference)


    def __call__(self,
                env: ManagerBasedRLEnv,
                object_cfg: SceneEntityCfg,
                grasping_reference_path: str,
                contact_key: str,
                thumb_rot_cfgs: SceneEntityCfg,
                thumb_flex_cfgs: SceneEntityCfg,
                thumb_finray_cfgs: SceneEntityCfg,
                right_flex_cfgs: SceneEntityCfg,
                right_finray_cfgs: SceneEntityCfg,
                left_flex_cfgs: SceneEntityCfg,
                left_finray_cfgs: SceneEntityCfg,
                threshold: float,
                position_threshold: float):

        
        obj_pos_w = get_obj_pos_w(env, self.object_cfg)
        obj_quat_w = get_obj_quat_w(env, self.object_cfg)
        
        
        w_pos_obj, w_quat_obj = subtract_frame_transforms(
            obj_pos_w, obj_quat_w
        )
        
        l_contact_pos_obj = []
        l_indicator_contact = []
        for key in self.t_contact_terms_keys:
            contact_sensor = self.d_contact_sensors[key]
            contact_pos_w = contact_sensor.data.contact_pos_w.squeeze(-2, -3)  # (num_envs, num_contact_sensors, 3)
            
            l_contact_pos_obj.append(subtract_frame_transforms(obj_pos_w, obj_quat_w, contact_pos_w)[0])
            l_indicator_contact.append(torch.where(torch.norm(contact_sensor.data.force_matrix_w[:, :, 0], dim=-1) > threshold, 1.0, 0.0))
            
            
        contact_pos_obj = torch.stack(l_contact_pos_obj, dim=1)  # (num_envs, num_contact_sensors, 3)
        indicator_contact = torch.stack(l_indicator_contact, dim=1).squeeze(-1)  # (num_envs, num_contact_sensors)
        
        norm_pos = torch.norm(contact_pos_obj - self.contact_ref_pos_obj,dim=-1)
        indicator_desired_contact_vector = torch.where(norm_pos < position_threshold, 1.0, 0.0)  # (num_envs, num_contact_sensors)
        
        first_term_reward_contact = torch.linalg.vecdot(indicator_contact, indicator_desired_contact_vector, dim=1)/ torch.linalg.vecdot(indicator_desired_contact_vector, indicator_desired_contact_vector, dim=1)  # (num_envs, )
    
        return first_term_reward_contact