from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": False})

import carb
from isaacsim.core.api import World
from scripts.iiwa_robot_test.tasks.follow_target import FollowTarget
from scripts.iiwa_robot_test.ik_solver import KinematicsSolver
from scripts.iiwa_robot_test.controllers.rmpflow import RMPFlowController
import numpy as np

my_world = World(stage_units_in_meters=1.0)
#Initialize the Follow Target task with a target location for the cube to be followed by the end effector
my_task = FollowTarget(name="iiwa_follow_target", target_position=np.array([0.5, 0, 0.5]))
my_world.add_task(my_task)
my_world.reset()
task_params = my_world.get_task("iiwa_follow_target").get_params()
target_name = task_params["target_name"]["value"]
denso_name = task_params["robot_name"]["value"]
my_denso = my_world.scene.get_object(denso_name)
#initialize the controller
# my_controller = KinematicsSolver(my_denso)

#initialize the controller
my_controller = RMPFlowController(name="target_follower_controller", robot_articulation=my_denso)
my_controller.reset()

articulation_controller = my_denso.get_articulation_controller()
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()
            my_controller.reset()
        observations = my_world.get_observations()
        actions = my_controller.forward(
            target_end_effector_position=observations[target_name]["position"],
            target_end_effector_orientation=observations[target_name]["orientation"],
        )
        articulation_controller.apply_action(actions)
        print(actions.joint_positions.round(4))
        # actions, succ = my_controller.compute_inverse_kinematics(
        #     target_position=observations[target_name]["position"],
        #     target_orientation=observations[target_name]["orientation"],
        # )
        # if succ:
        #     articulation_controller.apply_action(actions)
        # else:
        #     carb.log_warn("IK did not converge to a solution.  No action is being taken.")
simulation_app.close()