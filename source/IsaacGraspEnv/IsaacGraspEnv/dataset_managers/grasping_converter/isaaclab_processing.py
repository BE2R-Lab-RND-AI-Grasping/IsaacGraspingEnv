from typing import Any
import torch
import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext


from IsaacGraspEnv.robots.iiwa_cringe.dpflex_hand_free_cfg import DP_FLEX_FREE_CFG  # isort:skip
from pathlib import Path
import numpy as np
import inspect


class IsaacProcessingDataset():
    
    def __init__(self, args_cli, articulation_cfg, d_preprocessing_data_functions: dict[str, Any], l_postprocessing_data_funcions: list[Any]) -> None:
        
        self.d_preproc_func = d_preprocessing_data_functions
        self.l_postproc_func = l_postprocessing_data_funcions
        self.articulation_cfg = articulation_cfg
        
        sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
        self.sim = SimulationContext(sim_cfg)
        
        
    def _build_scene(self, num_envs: int, env_spaces: float):

        """Builds the scene."""
        # Ground-plane
        cfg = sim_utils.GroundPlaneCfg()
        cfg.func("/World/defaultGroundPlane", cfg)
        # Lights
        cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        cfg.func("/World/Light", cfg)

        origin_counter = 0
        x_offset = (num_envs // 2) * env_spaces
        y_offset = (num_envs // 2) * env_spaces
        x_values = np.linspace(-x_offset, x_offset, num=int(np.sqrt(num_envs)))
        y_values = np.linspace(-y_offset, y_offset, num=int(np.sqrt(num_envs)))
        origins = []

        for i in range(int(np.sqrt(num_envs))):   
            for j in range(int(np.sqrt(num_envs))):
                origins.append([x_values[i], y_values[j], 1.0])
                prim_utils.create_prim(f"/World/Origin{origin_counter}", "Xform", translation=origins[-1])
                origin_counter += 1
                
                if origin_counter >= num_envs:
                    break
            if origin_counter >= num_envs:
                    break


        articulation_cfg = self.articulation_cfg.copy()
        articulation_cfg.spawn.rigid_props.angular_damping = 100.0
        articulation_cfg.spawn.rigid_props.linear_damping = 100.0
        articulation_cfg.prim_path = "/World/Origin.*/Robot"
        articulation = Articulation(cfg=articulation_cfg)

        scene_entities = {"articulation": articulation}
        
        return scene_entities, origins
    
    
    def _preprocess_dataset(self,dataset: np.ndarray, *args, **kwargs) -> dict[str, torch.Tensor]:
        torch_dataset = {}
        for key, func in self.d_preproc_func.items():
            torch_dataset[key] = func(dataset, *args, **kwargs).to(self.sim.device)

        return torch_dataset
    
    
    def _run_step_simulation(self, entities: dict[str, Articulation], dataset: np.ndarray, *args, **kwargs) -> np.ndarray:
        
        # reset simulation to ensure a clean state
        
        self.sim.reset()
        
        articulations = entities["articulation"]
        
        torch_dataset = self._preprocess_dataset(dataset, *args, **kwargs)
        
        d_articulation_methods = dict(inspect.getmembers(articulations, predicate=inspect.ismethod))
        for key, value in torch_dataset.items():
            if key in d_articulation_methods:
                d_articulation_methods[key](value)
            else:
                raise KeyError(f"Method {key} not found in articulation methods {list(d_articulation_methods.keys())}")

        articulations.reset()
        articulations.write_data_to_sim()
        self.sim.step()
        articulations.update(self.sim.get_physics_dt())
        
        for func in self.l_postproc_func:
            func(articulations, dataset, *args, **kwargs)
                
        sim_dt = self.sim.get_physics_dt()
        
        return dataset


def make_torch_wrist_state(dataset: np.ndarray, *args, **kwargs) -> torch.Tensor:
    
    """Sets the wrist state of the robot."""
    origins = kwargs.get("origins", torch.zeros((dataset.shape[0], 3), device=kwargs.get("device", "cpu")))
    wrist_pos = torch.tensor([data["wrist_pos"] for data in dataset], device=origins.device) + origins
    wrist_quat = torch.tensor([data["wrist_quat"] for data in dataset], device=origins.device)
    wrist_state = torch.cat((wrist_pos, wrist_quat), dim=1)
    
    return wrist_state

def make_torch_joint_pos(dataset: np.ndarray, *args, **kwargs) -> torch.Tensor:
    
    """Sets the joint positions of the robot."""
    joint_order = kwargs.get("joint_order", None)
    if joint_order is None:
        raise ValueError("Joint Order must be provided in kwargs")
    
    joint_pos = torch.tensor([[data[joint_name] for joint_name in joint_order] for data in dataset], device=kwargs.get("device", "cpu"))

    return joint_pos

def make_zero_joint_vel(dataset: np.ndarray, *args, **kwargs) -> torch.Tensor:
    
    """Sets the joint velocities of the robot to zero."""
    joint_order = kwargs.get("joint_order", None)
    if joint_order is None:
        raise ValueError("Joint Order must be provided in kwargs")
    
    joint_vel = torch.zeros((dataset.shape[0], len(joint_order)), device=kwargs.get("device", "cpu"))

    return joint_vel

def make_zero_root_vel(dataset: np.ndarray, *args, **kwargs) -> torch.Tensor:
    
    """Sets the root velocities of the robot to zero."""
    root_vel = torch.zeros((dataset.shape[0], 6), device=kwargs.get("device", "cpu"))

    return root_vel
        
        
def log_bodies_pose(articulation: Articulation, dataset: np.ndarray, *args, **kwargs) -> None:
    """Logs the bodies pose of the robot."""
    origins = kwargs.get("origins", torch.zeros((dataset.shape[0], 3), device=kwargs.get("device", "cpu")))
    bodies_pos = articulation.data.body_link_pos_w.cpu().numpy()
    bodies_quat = articulation.data.body_link_quat_w.cpu().numpy()
    
    np_origins = origins.cpu().numpy()
    body_names = articulation.body_names
    
    for robot_id in range(origins.shape[0]):
        d_bodies_pose = {}
        for name, pos, quat in zip(body_names, bodies_pos[robot_id], bodies_quat[robot_id]):
            d_bodies_pose[name] = (pos - np_origins[robot_id], quat)
    
        dataset[robot_id]["bodies_pose"] = d_bodies_pose
