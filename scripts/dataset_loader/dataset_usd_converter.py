


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=True, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--dataset_path",
    type=str,
    help="Absolute path to dataset. Dataset directory must have folders with models.",
)
parser.add_argument(
    "--name_point_cloud",
    type=str,
    default=None,
    help="Define the name of the file containing the object's point cloud",
)
parser.add_argument(
    "--obj_file_name",
    type=str,
    help="Name of file containing object's mesh",
)
parser.add_argument(
    "--path_usd",
    type=str,
    help="Absolute path to created usd.",
)
#TODO: Create yaml parrser for inertial parameters of objects 
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import shutil
import contextlib
from isaaclab.sim.converters import MeshConverterCfg
from IsaacGraspEnv.sim.converters import ExtentedMeshConverter
from isaaclab.sim.schemas import schemas_cfg
from IsaacGraspEnv.sim.schemas import schemas_cfg as ext_schemas_cfg

from scipy.spatial.transform import Rotation as R


import pathlib

import carb
import isaacsim.core.utils.stage as stage_utils
import omni.kit.app

MASS_PROPERTIES_OBJECTS = {
    "power_drills": ("mass", 0.5),
    "wrenches": ("density", 7850),
    "combinational_wrenches": ("density", 7850),
    "screwdrivers": ("density", 7850),
    "locking_pliers": ("density", 7850),
    "fixed_joint_pliers": ("density", 7850),
}

def get_dir_paths_to_obj(path_to_dataset, file_obj, include_mdls=[], ):
    path_to_dataset = pathlib.Path(path_to_dataset)
    obj_directories = [dir.name for dir in path_to_dataset.iterdir() if dir.is_dir()]
    
    dirs_with_obj = []
    for o_dir in obj_directories:
        path_to_obj = path_to_dataset / o_dir
        checking_dir = lambda dir: dir.is_dir() and (path_to_obj /dir / file_obj).is_file()
        if include_mdls:
            checking_dir = lambda dir: dir.is_dir() and dir.name in include_mdls and (path_to_obj /dir / file_obj).is_file()

        model_directories = [dir.name for dir in path_to_obj.iterdir() if checking_dir(dir)]
        dirs_with_obj += model_directories
        
        return dirs_with_obj


def main():
    path_to_dataset = args_cli.dataset_path

    path_to_dataset = pathlib.Path(path_to_dataset)
    obj_classes = [dir.name for dir in path_to_dataset.iterdir() if dir.is_dir()]
    dict_db_structure = {}
    for o_cls in obj_classes:
        path_to_obj_cls = path_to_dataset / o_cls
        list_obj_dir_models = [dir.name for dir in path_to_obj_cls.iterdir() if dir.is_dir() and dir.name.split("_")[-1].isdigit()]
        dict_db_structure[o_cls] = sorted(list_obj_dir_models, key=lambda x: int(x.split("_")[-1]))


    path_usd_dataset = args_cli.path_usd#"/home/yefim-home/Documents/work/IsaacGraspingEnv/source/IsaacGraspEnv/IsaacGraspEnv/assets/data/HANDEL"

    path_usd_dataset = pathlib.Path(path_usd_dataset)

    for o_cls, models in dict_db_structure.items():
        mass_prop_obj = MASS_PROPERTIES_OBJECTS.get(o_cls, None)
        if mass_prop_obj:
            for obj_model in models:
                path_to_obj = path_to_dataset / o_cls / obj_model / args_cli.obj_file_name #'object_convex_decomposition_meter_unit.obj'

                # TODO: Remove Copy Labeled PC
                path_to_pc_labeled = path_to_dataset / o_cls / obj_model / 'point_cloud_labeled.ply'
                
                
                path_to_usd_dir = path_usd_dataset / o_cls / obj_model
                path_to_usd_file = path_to_usd_dir / "object.usd"
                
                
                mass_props = schemas_cfg.MassPropertiesCfg()
                setattr(mass_props, mass_prop_obj[0], mass_prop_obj[1])
                rigid_props = schemas_cfg.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=3666.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=192,
                    solver_velocity_iteration_count=1,
                    max_contact_impulse=1e32,
                )
                sdf_collision_props = ext_schemas_cfg.SDFCollisionPropertiesCfg(sdf_bits_per_subgrid_pixel="BitsPerPixel16")
                collision_props = ext_schemas_cfg.ExtendedCollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0001, sdf_collision_properties_cfg=sdf_collision_props)
                # collision_props = schemas_cfg.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0)

                mesh_converter_cfg = MeshConverterCfg(
                                mass_props=mass_props,
                                rigid_props=rigid_props,
                                collision_props=collision_props,
                                asset_path=str(path_to_obj),
                                collision_approximation = "sdf",
                                force_usd_conversion=True,
                                make_instanceable=True,
                                usd_dir = str(path_to_usd_dir), 
                                usd_file_name = str(path_to_usd_file),
                                # translation = (0.0, 0.05, 0.0),
                                translation = (0.0, 0.0, 0.0),
                                rotation = R.from_euler('xyz', [0.0, 0.0, 0.0],  degrees=True).as_quat()[[3,0,1,2]].tolist(),
                                )

                mesh_converter = ExtentedMeshConverter(mesh_converter_cfg, geom_name=f"{o_cls}_{obj_model}")
                
                if args_cli.name_point_cloud:
                    path_to_pc = path_to_dataset / o_cls / obj_model / args_cli.name_point_cloud
                    shutil.copy2(path_to_pc, path_to_usd_dir)
                shutil.copy2(path_to_pc_labeled, path_to_usd_dir)
                
                
    # Determine if there is a GUI to update:
    # acquire settings interface
    carb_settings_iface = carb.settings.get_settings()
    # read flag for whether a local GUI is enabled
    local_gui = carb_settings_iface.get("/app/window/enabled")
    # read flag for whether livestreaming GUI is enabled
    livestream_gui = carb_settings_iface.get("/app/livestream/enabled")

    # Simulate scene (if not headless)
    if local_gui or livestream_gui:
        # Open the stage with USD
        stage_utils.open_stage(mesh_converter.usd_path)
        # Reinitialize the simulation
        app = omni.kit.app.get_app_interface()
        # Run simulation
        with contextlib.suppress(KeyboardInterrupt):
            while app.is_running():
                # perform step
                app.update()


if __name__=="__main__":
    main()
    # close sim app
    simulation_app.close()