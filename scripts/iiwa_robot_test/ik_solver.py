from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from isaacsim.core.prims import Articulation
from typing import Optional


class KinematicsSolver(ArticulationKinematicsSolver):
    def __init__(self, robot_articulation: Articulation, end_effector_frame_name: Optional[str] = None) -> None:
        self._kinematics = LulaKinematicsSolver(robot_description_path="/home/yefim-home/Documents/work/IsaacGraspingEnv/scripts/iiwa_robot_test/rmpflow/robot_dexcriptor.yaml",
                                                urdf_path="/home/yefim-home/Documents/robots_models/iiwa_cringe_release/iiwa_cringe.urdf")
        if end_effector_frame_name is None:
            end_effector_frame_name = "lbr_iiwa_link_7"
        ArticulationKinematicsSolver.__init__(self, robot_articulation, self._kinematics, end_effector_frame_name)
        return