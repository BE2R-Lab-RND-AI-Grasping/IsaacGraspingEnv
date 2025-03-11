
from __future__ import annotations

import torch
from collections.abc import Sequence
from omni.isaac.lab.managers import SceneEntityCfg
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

def set_pd_offset(env: ManagerBasedRLEnv,
                  env_ids: Sequence[int],
                  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                  pd_offset: float = 0.0,
                  ):

    arr_size = env.scene["robot"].data.joint_pos_target[:, robot_cfg.joint_ids].size()
    
    tensor_pd_offset = (torch.ones(arr_size) * pd_offset).to(env.sim.device)
    
    env.scene["robot"].data.joint_pos_target[:, robot_cfg.joint_ids] = tensor_pd_offset
    