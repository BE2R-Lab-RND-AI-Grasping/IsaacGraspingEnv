from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg


from scipy.spatial.transform import Rotation as R
import numpy as np
import pathlib

OBJ_POS = np.array([0.9, 0.0, 0.07077])
# OBJ_POS = np.array([0.0, 0.0, 10.0])
OBJ_ROT = R.from_euler("xyz", [90.0, 0.0, 0.0], degrees=True)

def get_filtered_folder_name_in_dirs(path_to_models, models_filter):
    dir_check = lambda dir: dir.is_dir() and dir.name.split("_")[-1].isdigit()
    if models_filter:
        dt_filter = lambda dir: dir.name.split("_")[-1] in models_filter
    else:
        dt_filter = lambda dir: True
        
    list_folder_names = [
        dir.name
        for dir in path_to_models.iterdir()
        if dir_check(dir) and dt_filter(dir)
    ]
    list_folder_names = sorted(list_folder_names, key=lambda x: int(x.split("_")[-1]))

    return list_folder_names


def load_object_dataset(path_to_models, usd_file_name, models_filter=None):
    path_to_models = pathlib.Path(path_to_models)
    
    list_obj_models = get_filtered_folder_name_in_dirs(path_to_models, models_filter)
    
    num_obj_models = len(list_obj_models)
    dict_obj_models = {}
    for o_model in list_obj_models:
        dict_obj_models[o_model] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object" + f"_{o_model}",
            # From Front
            # init_state=RigidObjectCfg.InitialStateCfg(pos=[0.9, 0, 0.08], rot=[0.    , 0.    , 0.7071, 0.7071]),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=OBJ_POS.tolist(), rot=OBJ_ROT.as_quat()[[3, 0, 1, 2]].tolist()
            ),
            # From Top
            # init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                # usd_path=f"source/IsaacGraspEnv/IsaacGraspEnv/assets/data/YCB/035_power_drill_wo_texture.usd",
                usd_path=str(path_to_models / o_model / usd_file_name),
                # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/035_power_drill.usd",
                # usd_path=f"source/IsaacGraspEnv/IsaacGraspEnv/assets/data/YCB/Props/instanceable_meshes.usd",
                scale=(1.0, 1.0, 1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.001, rest_offset=0.0001
                ),
                semantic_tags=[("class", "object"), ("color", "red")],
            ),
        )

    return dict_obj_models


def preprocessing_point_cloud_loading(path_to_models, pc_file_name, models_filter=None):

    path_to_models = pathlib.Path(path_to_models)
    list_obj_models = get_filtered_folder_name_in_dirs(path_to_models, models_filter)
    
    
    list_path_to_pc = []
    for o_model in list_obj_models:
        
        list_path_to_pc.append(str(path_to_models / o_model / pc_file_name))
        
    return list_path_to_pc
    
    