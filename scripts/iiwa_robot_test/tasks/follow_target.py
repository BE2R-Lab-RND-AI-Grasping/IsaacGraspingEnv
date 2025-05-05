from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.core.utils.stage import add_reference_to_stage
import isaacsim.core.api.tasks as tasks
from typing import Optional
import numpy as np


# Inheriting from the base class Follow Target
class FollowTarget(tasks.FollowTarget):
    def __init__(
        self,
        name: str = "iiwa_follow_target",
        target_prim_path: Optional[str] = None,
        target_name: Optional[str] = None,
        target_position: Optional[np.ndarray] = None,
        target_orientation: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        tasks.FollowTarget.__init__(
            self,
            name=name,
            target_prim_path=target_prim_path,
            target_name=target_name,
            target_position=target_position,
            target_orientation=target_orientation,
            offset=offset,
        )
        return

    def set_robot(self) -> SingleManipulator:
        asset_path = "/home/yefim-home/Documents/work/IsaacGraspingEnv/source/IsaacGraspEnv/IsaacGraspEnv/robots/hand_iiwa/hand_iiwa.usd"
        add_reference_to_stage(usd_path=asset_path, prim_path="/World/hand_iiwa")
        gripper = ParallelGripper(
            #We chose the following values while inspecting the articulation
            end_effector_prim_path="/World/hand_iiwa/lbr_iiwa_link_7",
            joint_prim_names=["Joint_thumb_dynamixel_crank", "Joint_right_dynamixel_crank"],
            joint_opened_positions=np.array([0, 0]),
            joint_closed_positions=np.array([2.3/2, 2.3/2]),
            action_deltas=np.array([2.3/2, 2.3/2]),
        )
        manipulator = SingleManipulator(prim_path="/World/hand_iiwa",
                                        name="hand_iiwa",
                                        end_effector_prim_name="lbr_iiwa_link_7",
                                        gripper=gripper)
        INIT_Q_IIWA = np.array([0.0, 71.8, 0.0, -89.3, 0.0, -71.8, 0.0])/180*np.pi
        INIT_Q_IIWA = np.hstack([INIT_Q_IIWA, np.zeros(16)])
        INIT_Q_IIWA = INIT_Q_IIWA.tolist()
        manipulator.set_joints_default_state(positions=INIT_Q_IIWA)
        return manipulator