

from typing import Any, Optional
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from source.IsaacGraspEnv.IsaacGraspEnv.dataset_managers.grasping_converter.base import DatasetGraspingConverterBuilder


def position_processing_func(grasp: dict[str, Any], key: list[str]) -> list[float]:
    return [grasp[k] for k in key]

def orientation_processing_func(grasp: dict[str, Any], key: list[str]) -> list[float]:
    r = R.from_euler("xyz", [grasp[k] for k in key], degrees=False)
    return r.as_quat()[[3, 0, 1, 2]].tolist()



class DexGraspNetConverter():
    
    
    def __init__(self) -> None:
        
        self.dataset_paths = []
        self.new_path: Optional[Path] = None
        self.connected_objects = []
        self.dict_map_keys = {}
        self.processing_functions = {}
    
    def convert_grasping_one_object(self, dataset: np.ndarray) -> np.ndarray:
        new_grasp_datset = []
        for grasp in dataset:
            new_grasp = {}
            for key, value in self.dict_map_keys.items():
                new_grasp[key] = self.processing_functions[key](grasp, value) if value in self.processing_functions else grasp[value]
            new_grasp_datset.append(new_grasp)
            
        return np.array(new_grasp_datset)
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        
        for dir_path in self.dataset_paths:
            print(f"Processing dataset in path: {dir_path}")
            if not dir_path.exists():
                raise FileNotFoundError(f"Dataset path {dir_path} does not exist.")
            
            l_file_paths = list(dir_path.glob("*.npy"))
            
            for path in l_file_paths:
                with open(path, "rb") as f:
                    data = np.load(f, allow_pickle=True)
                    
                new_data = self.convert_grasping_one_object(data)
                
                with open(self.new_path / path.name, "wb") as f:
                    np.save(f, new_data)


class DexGraspNetConverterBuilder(DatasetGraspingConverterBuilder):
    def __init__(self) -> None:
        self._converter = DexGraspNetConverter()
        
    @property
    def converter(self) -> DexGraspNetConverter:
        return self._converter

    def define_dataset_paths(self, old_paths: list[str], new_path: str) -> None:
        self._converter.dataset_paths = [Path(path) for path in old_paths]
        self._converter.new_path = Path(new_path)

    def define_connected_objects(self, paths_to_object: list[str]) -> None:
        self._converter.connected_objects = [Path(path) for path in paths_to_object]

    def define_dataset_structure(self, dict_map_keys: dict[str, str]) -> None:
        self._converter.dict_map_keys = dict_map_keys
        
