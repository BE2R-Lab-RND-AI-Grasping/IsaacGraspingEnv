"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)

# append AppLauncher cli args

# parser.add_argument(
#     "--dataset_path",
#     type=str,
#     default="/home/yefim-home/Documents/work/IsaacGraspingEnv/source/IsaacGraspEnv/IsaacGraspEnv/assets/data/HANDEL/locking_pliers",
#     help="Absolute path to dataset. Dataset directory must have folders with models.",
# )
# parser.add_argument(
#     "--usd_file_name",
#     type=str,
#     default="object.usd",
#     help="The name of the USD file in the folder",
# )
# parser.add_argument(
#     "--model_filter",
#     type=str,
#     default="1",
#     help="A comma separated list of identifiers to be taken from the dataset",
# )

# parser.add_argument(
#     "--grasp_dataset_path",
#     type=str,
#     default="/home/yefim-home/Documents/work/repo_forks/DexGraspNet/grasp_generation_egorhand_edited_hand/ready_to_work/dataset/DIP-Flex_opened_kinematics",
#     help="Absolute path to dataset. Dataset directory must have folders with models.",
# )
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import source.IsaacGraspEnv.IsaacGraspEnv.dataset_managers.grasping_converter as gc


def main():

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


if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()
