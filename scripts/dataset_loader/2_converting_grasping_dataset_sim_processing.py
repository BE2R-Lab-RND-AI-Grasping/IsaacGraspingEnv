import argparse

from isaaclab.app import AppLauncher
from pathlib import Path

parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")


parser.add_argument(
    "--env_spaces", type=int, default=0.5, help="Size of the environment spaces."
)

parser.add_argument(
    "--max_size_dataset",
    type=int,
    default=100,
    help="Max size of the dataset to be processed.",
)

AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import numpy as np

from IsaacGraspEnv.dataset_managers.grasping_converter import (
    IsaacProcessingDataset,
    make_torch_wrist_state,
    make_torch_joint_pos,
    make_zero_joint_vel,
    make_zero_root_vel,
    log_bodies_pose,
)
from IsaacGraspEnv.robots.iiwa_cringe.dpflex_hand_free_cfg import (
    DP_FLEX_FREE_CFG,
)  # isort:skip


d_preprocessing_data_functions = {
    "write_root_pose_to_sim": make_torch_wrist_state,
    "write_joint_position_to_sim": make_torch_joint_pos,
    "write_joint_velocity_to_sim": make_zero_joint_vel,
    "write_root_velocity_to_sim": make_zero_root_vel,
}

l_postprocessing_data_functions = [
    log_bodies_pose,
]
# Path to dataset files to be processed
PATHS = []


def main():

    isaac_dataset_processing = IsaacProcessingDataset(
        args_cli.max_size_dataset,
        args_cli.device,
        args_cli.env_spaces,
        DP_FLEX_FREE_CFG,
        d_preprocessing_data_functions,
        l_postprocessing_data_functions,
    )

    for path in PATHS:
        paths_to_file = Path(path).rglob("*.npy")
        for file_path in paths_to_file:
            with open(file_path, "rb") as f:
                dataset = np.load(f, allow_pickle=True)

            new_dataset = isaac_dataset_processing(dataset)

            with open(file_path, "wb") as f:
                np.save(f, new_dataset)


if __name__ == "__main__":
    main()

    simulation_app.close()
