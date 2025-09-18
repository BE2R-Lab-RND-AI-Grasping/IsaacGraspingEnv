

from typing import Any, Optional
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from source.IsaacGraspEnv.IsaacGraspEnv.dataset_managers.grasping_converter.base import DatasetGraspingConverterBuilder
from source.IsaacGraspEnv.IsaacGraspEnv.dataset_managers.grasping_converter.isaaclab_processing import IsaacProcessingDataset

def unpack_including_dict(key_unpacked_dict: str):
    def decorator(func):
        def wrapper(grasp: dict[str, Any], keys: list[str]) -> Any:
            if key_unpacked_dict not in grasp:
                result = func(grasp, keys)
            else:
                unpacked_dict = grasp[key_unpacked_dict]
                result = func(unpacked_dict, keys)
            return result
        return wrapper
    return decorator

@unpack_including_dict("qpos")
def position_processing_func(grasp: dict[str, Any], keys: list[str]) -> list[float]:
    return [grasp[k] for k in keys]


@unpack_including_dict("qpos")
def orientation_rpy2quat_processing_func(grasp: dict[str, Any], keys: list[str]) -> list[float]:
    r = R.from_euler("xyz", [grasp[k] for k in keys], degrees=False)
    return r.as_quat()[[3, 0, 1, 2]].tolist()


def convert_name_dexgraspnet(object_name: str, model_name: str, old_name: str) -> str:
    
    if old_name.find("_") > 0:
        splitter = "_"
    elif old_name.find("-") > 0:
        splitter = "-"
    else:
        splitter = " "
                
    dataset_id = old_name.split(".")[0].split(splitter)[-1]
    new_dataset_name = f"{model_name}_{dataset_id}.npy"
    
    return new_dataset_name


class DexGraspNetConverter():
    
    
    def __init__(self) -> None:
        
        self.dataset_paths: list[Path] = []
        self.new_path: Path = Path()
        self.d_filter_by_obj_re: dict[Path, str] = {}
        self.dict_map_keys: dict[str, str | list[str]] = {}
        self.d_proc_structure_func: dict[str, Any] = {}
        self.list_filter_file_re: list[str] = []
        self.func_convert_name: Optional[Any] = None
    
    def convert_grasping_one_object(self, dataset: np.ndarray) -> np.ndarray:
        new_grasp_datset = []
        for grasp in dataset:
            new_grasp = {}
            for key, value in self.dict_map_keys.items():
                new_grasp[key] = self.d_proc_structure_func[key](grasp, value) if key in self.d_proc_structure_func else grasp["qpos"][value]
            new_grasp_datset.append(new_grasp)
            
        return np.array(new_grasp_datset)
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        
        for dir_path in self.dataset_paths:
            print(f"Processing dataset in path: {dir_path}")
            if not dir_path.exists():
                raise FileNotFoundError(f"Dataset path {dir_path} does not exist.")
            
            s_file_paths = set()
            for fltr_files_re in self.list_filter_file_re:
                s_file_paths = s_file_paths.union(set(dir_path.glob(fltr_files_re)))
            
            for path, fltr_obj_re in self.d_filter_by_obj_re.items():
                
                fltred_file_by_obj = set(dir_path.glob(fltr_obj_re))
                fltred_file_by_obj = fltred_file_by_obj.intersection(s_file_paths)
                
                object_name = path.parents[0].name
                model_name = path.name
                
                path_to_new_data_dir = self.new_path / object_name
                
                
                if not (path_to_new_data_dir).exists():
                    path_to_new_data_dir.mkdir(parents=True, exist_ok=True)
                
                for dataset_path in fltred_file_by_obj:    
                    with open(dataset_path, "rb") as f:
                        data = np.load(f, allow_pickle=True)
                        
                    new_data = self.convert_grasping_one_object(data)
                    
                    if self.func_convert_name is not None:
                        new_dataset_name = self.func_convert_name(object_name, model_name, dataset_path.name)
                    else:
                        new_dataset_name = dataset_path.name
                    
                    with open(path_to_new_data_dir / new_dataset_name, "wb") as f:
                        np.save(f, new_data, allow_pickle=True)


class DexGraspNetConverterBuilder(DatasetGraspingConverterBuilder):
    def __init__(self) -> None:
        self._converter = DexGraspNetConverter()
        
    @property
    def converter(self) -> DexGraspNetConverter:
        return self._converter

    def define_dataset_paths(self, old_paths: list[str], new_path: str) -> None:
        self._converter.dataset_paths = [Path(path) for path in old_paths]
        self._converter.new_path = Path(new_path)

    def define_connected_objects(self, paths_to_object: dict[str, str]) -> None:
        self._converter.d_filter_by_obj_re = {Path(path): reg_expr for path, reg_expr in paths_to_object.items()}

    def define_dataset_structure(self, dict_joint_key_map: dict[str, str]) -> None:
        l_trans_keys = ["WRJTx", "WRJTy", "WRJTz"]
        l_rpy_rot_keys = ["WRJRx", "WRJRy", "WRJRz"]
        
        d_key_maps: dict[str, str | list[str]] = {}
        d_key_maps["wrist_pos"] =  l_trans_keys
        d_key_maps["wrist_quat"] =  l_rpy_rot_keys
        
        d_key_maps.update(dict_joint_key_map)
        
        self._converter.dict_map_keys = d_key_maps
        
    def define_dataset_structure_processing(self, processing_functions: Optional[dict[str, Any]] = None) -> None:
        
        d_proc_func: dict[str, Any] = {}
        d_proc_func["wrist_pos"] = position_processing_func
        d_proc_func["wrist_quat"] = orientation_rpy2quat_processing_func
        
        if processing_functions is not None:
            d_proc_func.update(processing_functions)
        
        self._converter.d_proc_structure_func = d_proc_func
        
    def define_file_filter(self, list_re_filters: Optional[list[str]] = None) -> None:
        if list_re_filters is None or len(list_re_filters) == 0:
            list_re_filters = ["*.npy"]

        self._converter.list_filter_file_re = list_re_filters
        
    def define_name_conversion_function(self, func_convert_name: Optional[Any] = None) -> None:
        self._converter.func_convert_name = func_convert_name

        


if __name__ == "__main__":
    
    from source.IsaacGraspEnv.IsaacGraspEnv.dataset_managers.grasping_converter.utils import resolve_names_matching
    
    l_joint_names = ['Joint_left_abduction', 'Joint_right_abduction', 'Joint_thumb_rotation',
                    'Joint_left_flexion', 'Joint_right_flexion', 'Joint_thumb_abduction',
                    'Joint_left_finray_proxy', 'Joint_right_finray_proxy', 'Joint_thumb_flexion',
                    'Joint_thumb_finray_proxy']
    
    map_old2new_joint_names = {
    "left": "index",
    "right": "pinkie",
    "thumb": "thumb",
    "rotation": "rotation",
    "flexion": "PPflexion",
    "finray_proxy": "DPflexion",
    }
    
    old_paths = ["/home/yefim-home/Documents/work/repo_forks/DexGraspNet/grasp_generation_egorhand_edited_hand/ready_to_work/dataset/DIP-Flex_opened_kinematics"]
    new_path = "/home/yefim-home/Documents/work/IsaacGraspingEnv/datasets/dexgraspnet_converted"
    
    path_to_object = {
        "/home/yefim-home/Documents/work/IsaacGraspingEnv/source/IsaacGraspEnv/IsaacGraspEnv/assets/data/HANDEL/locking_pliers/model_0" : "*pliers*",
        "/home/yefim-home/Documents/work/IsaacGraspingEnv/source/IsaacGraspEnv/IsaacGraspEnv/assets/data/HANDEL/screwdrivers/model_0" : "*screwdriver*",
    }
    
    dict_map_keys = resolve_names_matching(l_joint_names, map_old2new_joint_names)
    
    conveter_builder = DexGraspNetConverterBuilder()
    conveter_builder.define_dataset_paths(old_paths, new_path)
    conveter_builder.define_connected_objects(path_to_object)
    conveter_builder.define_dataset_structure(dict_map_keys)
    conveter_builder.define_dataset_structure_processing()
    conveter_builder.define_file_filter()
    conveter_builder.define_name_conversion_function(convert_name_dexgraspnet)
    
    converter = conveter_builder.converter
    converter()