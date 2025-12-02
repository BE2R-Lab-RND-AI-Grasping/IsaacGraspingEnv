
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
import yaml

class DatasetGraspingConverterBuilder(ABC):
    """
    
    """

    @property
    @abstractmethod
    def converter(self) -> None:
        pass
    

    @abstractmethod
    def define_dataset_paths(self, old_paths: list[str], new_path: str) -> None:
        pass

    @abstractmethod
    def define_connected_objects(self, paths_to_object: list[str]) -> None:
        pass

    @abstractmethod
    def define_dataset_structure(self, dict_map_keys: dict[str, str]) -> None:
        pass
    
    @abstractmethod
    def define_dataset_structure_processing(self, processing_functions: dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def define_file_filter(self, list_re_filters: list[str]) -> None:
        pass
    
    
    def define_converter_by_yaml_config(self, path_to_config: Path) -> None:
        with open(path_to_config, "r") as f:
            dict_config = yaml.safe_load(f)
            
        self.define_dataset_paths(dict_config["dataset_paths"]["old"], dict_config["dataset_paths"]["new"])
        self.define_connected_objects(dict_config["connection_object_file_n_dataset"])
        self.define_dataset_structure(dict_config["dataset_structure"])
        
        list_str_file_filtering = dict_config["dataset_function_processing"]
        if list_str_file_filtering and len(list_str_file_filtering) > 0:
            self.define_file_filter(list_str_file_filtering)
