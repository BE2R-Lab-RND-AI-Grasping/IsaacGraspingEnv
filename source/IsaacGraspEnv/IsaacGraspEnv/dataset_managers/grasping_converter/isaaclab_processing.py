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

from functools import lru_cache


class IsaacProcessingDataset():
    
    def __init__(self, max_size_dataset, device, env_spaces, articulation_cfg, d_preprocessing_data_functions: dict[str, Any], l_postprocessing_data_funcions: list[Any]) -> None:
        """Processes a grasping dataset using Isaac Sim.
        
        Args:
            max_size_dataset (int): Maximum size of the dataset to process.
            device (str): Device to run the simulation on.
            env_spaces (float): Size of the environment spaces.
            articulation_cfg (ArticulationCfg): Configuration of the robot articulation.
            d_preprocessing_data_functions (dict[str, Any]): Dictionary of functions for preprocessing the dataset.
            l_postprocessing_data_functions (list[Any]): List of functions for postprocessing the dataset."""
        self.max_size_dataset = max_size_dataset
        self.d_preproc_func = d_preprocessing_data_functions
        self.l_postproc_func = l_postprocessing_data_funcions
        self.articulation_cfg = articulation_cfg
        
        sim_cfg = sim_utils.SimulationCfg(device=device)
        self.sim = SimulationContext(sim_cfg)

        self._entities, self._origins = self._build_scene(env_spaces=env_spaces)


    def _build_scene(self, env_spaces: float):
        """Builds the scene.
        Args:
            env_spaces (float): Size of the environment spaces.
        Returns:
            scene_entities (dict): Dictionary of scene entities.
            origins (list[list[float]]): List of origins of the environments.
        """
        # Ground-plane
        cfg = sim_utils.GroundPlaneCfg()
        cfg.func("/World/defaultGroundPlane", cfg)
        # Lights
        cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        cfg.func("/World/Light", cfg)


        origin_counter = 0
        x_offset = (self.max_size_dataset // 2) * env_spaces
        y_offset = (self.max_size_dataset // 2) * env_spaces
        x_values = np.linspace(-x_offset, x_offset, num=int(np.ceil(np.sqrt(self.max_size_dataset))))
        y_values = np.linspace(-y_offset, y_offset, num=int(np.ceil(np.sqrt(self.max_size_dataset))))
        origins = []

        for i in range(x_values.shape[0]):   
            for j in range(y_values.shape[0]):
                origins.append([x_values[i], y_values[j], 1.0])
                prim_utils.create_prim(f"/World/Origin{origin_counter}", "Xform", translation=origins[-1])
                origin_counter += 1
                
                if origin_counter >= self.max_size_dataset:
                    break
            if origin_counter >= self.max_size_dataset:
                    break


        articulation_cfg = self.articulation_cfg.copy()
        articulation_cfg.spawn.rigid_props.angular_damping = 100.0
        articulation_cfg.spawn.rigid_props.linear_damping = 100.0
        articulation_cfg.prim_path = "/World/Origin.*/Robot"
        articulation = Articulation(cfg=articulation_cfg)

        scene_entities = {"articulation": articulation}
        
        return scene_entities, origins
    
    
    def _preprocess_dataset(self,dataset: np.ndarray, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Preprocesses the dataset.
        
        Args:
            dataset (np.ndarray): Dataset to preprocess.
            *args: Additional arguments to pass to the preprocessing functions.
            **kwargs: Additional keyword arguments to pass to the preprocessing functions.
        Returns:
            torch_dataset (dict[str, torch.Tensor]): Preprocessed dataset.
        """
        torch_dataset = {}
        for key, func in self.d_preproc_func.items():
            torch_dataset[key] = func(dataset, *args, **kwargs).to(self.sim.device)

        return torch_dataset
    
    
    def _run_step_simulation(self, entities: dict[str, Articulation], dataset: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Runs a step of the simulation.
        
        Args:
            entities (dict[str, Articulation]): Dictionary of scene entities.
            dataset (np.ndarray): Dataset to process.
            *args: Additional arguments to pass to the preprocessing and postprocessing functions.
            **kwargs: Additional keyword arguments to pass to the preprocessing and postprocessing functions.
        Returns:
            dataset (np.ndarray): Processed dataset."""
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


    def __call__(self, dataset: np.ndarray, *args: Any, **kwds: Any) -> Any:
        """Processes the dataset.
        
        Args:
            dataset (np.ndarray): Dataset to process.
            *args: Additional arguments to pass to the preprocessing and postprocessing functions.
            **kwds: Additional keyword arguments to pass to the preprocessing and postprocessing functions.
        Returns:
            dataset (np.ndarray): Processed dataset.
        """
        # reset simulation to ensure a clean state
        self.sim.reset()

        dataset = self._run_step_simulation(self._entities, dataset, 
                                            origins=torch.tensor(self._origins, device=self.sim.device), 
                                            joint_order=self._entities["articulation"].joint_names, 
                                            device=self.sim.device, 
                                            num_envs=self.max_size_dataset, 
                                            *args, **kwds)

        return dataset


def make_torch_wrist_state(dataset: np.ndarray, *args, **kwargs) -> torch.Tensor:
    """Preprocessing: Sets the wrist state of the robot in simulation."""
    origins = kwargs.get("origins", torch.zeros((dataset.shape[0], 3), device=kwargs.get("device", "cpu")))
    wrist_pos = origins.clone()
    wrist_quat = torch.zeros((origins.shape[0], 4), device=kwargs.get("device", "cpu"))
    
    wrist_pos_dataset = torch.tensor([data["wrist_pos"] for data in dataset], device=origins.device)
    wrist_quat_dataset = torch.tensor([data["wrist_quat"] for data in dataset], device=origins.device)
    
    wrist_pos[:wrist_pos_dataset.shape[0], :3] += wrist_pos_dataset
    wrist_quat[:wrist_quat_dataset.shape[0], :] = wrist_quat_dataset

    wrist_state = torch.cat((wrist_pos, wrist_quat), dim=1)
    
    return wrist_state

def make_torch_joint_pos(dataset: np.ndarray, *args, **kwargs) -> torch.Tensor:
    """Preprocessing: Sets the joint positions of the robot in simulation."""
    joint_order = kwargs.get("joint_order", None)
    if joint_order is None:
        raise ValueError("Joint Order must be provided in kwargs")

    joint_pos_full = torch.zeros((kwargs.get("num_envs", dataset.shape[0]), len(joint_order)), device=kwargs.get("device", "cpu"))

    joint_pos = torch.tensor([[data.get(joint_name, 0.0) for joint_name in joint_order] for data in dataset], device=kwargs.get("device", "cpu"))

    joint_pos_full[:joint_pos.shape[0], :] = joint_pos

    return joint_pos_full

def make_zero_joint_vel(dataset: np.ndarray, *args, **kwargs) -> torch.Tensor:
    """Preprocessing: Sets the joint velocities of the robot in simulation."""
    joint_order = kwargs.get("joint_order", None)
    if joint_order is None:
        raise ValueError("Joint Order must be provided in kwargs")
    
    joint_vel = torch.zeros((kwargs.get("num_envs", dataset.shape[0]), len(joint_order)), device=kwargs.get("device", "cpu"))

    return joint_vel

def make_zero_root_vel(dataset: np.ndarray, *args, **kwargs) -> torch.Tensor:
    """Preprocessing: Sets the root velocities of the robot in simulation."""
    root_vel = torch.zeros((kwargs.get("num_envs", dataset.shape[0]), 6), device=kwargs.get("device", "cpu"))

    return root_vel
        
        
def log_bodies_pose(articulation: Articulation, dataset: np.ndarray, *args, **kwargs) -> None:
    """Postprocessing: Mutate dataset. Logs the poses of the bodies in the dataset."""
    origins = kwargs.get("origins", torch.zeros((dataset.shape[0], 3), device=kwargs.get("device", "cpu")))
    bodies_pos = articulation.data.body_link_pos_w.cpu().numpy()
    bodies_quat = articulation.data.body_link_quat_w.cpu().numpy()
    
    np_origins = origins.cpu().numpy()
    body_names = articulation.body_names
    
    for robot_id in range(dataset.shape[0]):
        d_bodies_pose = {}
        for name, pos, quat in zip(body_names, bodies_pos[robot_id], bodies_quat[robot_id]):
            d_bodies_pose[name] = (pos - np_origins[robot_id], quat)
    
        dataset[robot_id]["bodies_pose"] = d_bodies_pose
