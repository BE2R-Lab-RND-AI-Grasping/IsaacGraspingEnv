
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

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
