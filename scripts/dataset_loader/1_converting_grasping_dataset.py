"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")

AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import yaml
import source.IsaacGraspEnv.IsaacGraspEnv.dataset_managers.grasping_converter as gc


def main_manual():

    l_joint_names = [
        "Joint_left_abduction",
        "Joint_right_abduction",
        "Joint_thumb_rotation",
        "Joint_left_flexion",
        "Joint_right_flexion",
        "Joint_thumb_abduction",
        "Joint_left_finray_proxy",
        "Joint_right_finray_proxy",
        "Joint_thumb_flexion",
        "Joint_thumb_finray_proxy",
    ]

    map_old2new_joint_names = {
        "left": "index",
        "right": "pinkie",
        "thumb": "thumb",
        "rotation": "rotation",
        "flexion": "PPflexion",
        "finray_proxy": "DPflexion",
    }

    old_paths = [
        "/home/yefim-home/Documents/work/repo_forks/DexGraspNet/grasp_generation_egorhand_edited_hand/ready_to_work/dataset/DIP-Flex_opened_kinematics"
    ]
    new_path = "/home/yefim-home/Documents/work/IsaacGraspingEnv/datasets/dexgraspnet_converted"

    path_to_object = {
        "/home/yefim-home/Documents/work/IsaacGraspingEnv/source/IsaacGraspEnv/IsaacGraspEnv/assets/data/HANDEL/locking_pliers/model_0": "*pliers*",
        "/home/yefim-home/Documents/work/IsaacGraspingEnv/source/IsaacGraspEnv/IsaacGraspEnv/assets/data/HANDEL/screwdrivers/model_0": "*screwdriver*",
    }

    dict_map_keys = gc.resolve_names_matching(l_joint_names, map_old2new_joint_names)

    conveter_builder = gc.DexGraspNetConverterBuilder()
    conveter_builder.define_dataset_paths(old_paths, new_path)
    conveter_builder.define_connected_objects(path_to_object)
    conveter_builder.define_dataset_structure(dict_map_keys)
    conveter_builder.define_dataset_structure_processing()
    conveter_builder.define_file_filter()
    conveter_builder.define_name_conversion_function(gc.convert_name_dexgraspnet)

    converter = conveter_builder.converter
    converter()

    dict_config = {}

    dict_config["dataset_paths"] = {"old": old_paths, "new": new_path}
    dict_config["connection_object_file_n_dataset"] = path_to_object
    dict_config["dataset_structure"] = dict_map_keys
    dict_config["file_filtering"] = []

    yaml.safe_dump(
        dict_config, open("test.yaml", "w"), sort_keys=False, default_flow_style=False
    )


def main_config():

    from pathlib import Path

    conveter_builder = gc.DexGraspNetConverterBuilder()
    conveter_builder.define_converter_by_yaml_config(
        Path(
            "source/IsaacGraspEnv/IsaacGraspEnv/dataset_managers/grasping_converter/configs/dexgraspnet.yaml"
        )
    )
    conveter_builder.define_dataset_structure_processing()
    conveter_builder.define_name_conversion_function(gc.convert_name_dexgraspnet)

    converter = conveter_builder.converter
    converter()


if __name__ == "__main__":
    main_config()
    # close sim app
    simulation_app.close()
