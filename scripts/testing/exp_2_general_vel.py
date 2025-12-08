from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})


from pxr import UsdPhysics, PhysxSchema, Gf, PhysicsSchemaTools, UsdGeom, Usd
import omni
import numpy as np
from spatialmath import UnitQuaternion as UQ

import modern_robotics as mr
import omni.physics.tensors.impl.api as physx
import asyncio


async def rigid_body_abstraction(H_w0, H_w1):
    # wait a frame to ensure the physics simulation is initialized
    await omni.kit.app.get_app_interface().next_update_async()
    try:
        simulation_view = physx.create_simulation_view("torch")
        rigid_body1_view = simulation_view.create_rigid_body_view("/World/rigidBody1")
        rigid_body0_view = simulation_view.create_rigid_body_view("/World/rigidBody0")
        rb1_velocities = rigid_body1_view.get_velocities()
        rb1_velocities_np = rb1_velocities.numpy().reshape(rigid_body1_view.count, 6)

        rb0_velocities = rigid_body0_view.get_velocities()
        rb0_velocities_np = rb0_velocities.numpy().reshape(rigid_body0_view.count, 6)

        R_w0, pos_w0 = mr.TransToRp(H_w0)
        spatial_velocity_w_w0 = (
            -mr.VecToso3(rb0_velocities_np[0, 3:]) @ pos_w0 + rb0_velocities_np[0, :3]
        )
        twist_w_w0 = np.hstack([rb0_velocities_np[0, 3:], spatial_velocity_w_w0])

        R_w1, pos_w1 = mr.TransToRp(H_w1)
        spatial_velocity_w_w1 = (
            -mr.VecToso3(rb1_velocities_np[0, 3:]) @ pos_w1 + rb1_velocities_np[0, :3]
        )
        twist_w_w1 = np.hstack([rb1_velocities_np[0, 3:], spatial_velocity_w_w1])

        Ad_H_w0_inv = mr.Adjoint(mr.TransInv(H_w0))

        twist_0_0w = -Ad_H_w0_inv @ twist_w_w0
        # twist_0_0w = Ad_H_w0_inv @ twist_w_0w

        twist_0_01 = twist_0_0w + Ad_H_w0_inv @ twist_w_w1

        print(f"{'PhysX':=^20}")
        print(f"Origin Velocity: {rb1_velocities_np.round(4)}")
        print(f"Twist^0,0_1 ver 1: {twist_0_01.round(4)}")
        print(
            f"Origin Velocity from Spatial Vel ver 1: {(mr.VecTose3(twist_0_01) @ mr.TransInv(H_w0) @ np.hstack([pos_w1,[1]])).round(4)}"
        )
        print(f"{'PhysX':=^20}")
    except Exception as e:
        print(e)


def main():
    stage = omni.usd.get_context().get_stage()

    # Setting up Physics Scene
    gravity = 0.0
    scene = UsdPhysics.Scene.Define(stage, "/World/physics")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(gravity)
    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/World/physics"))
    physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/World/physics")
    physxSceneAPI.CreateEnableCCDAttr(True)
    physxSceneAPI.CreateEnableStabilizationAttr(True)
    physxSceneAPI.CreateEnableGPUDynamicsAttr(False)
    physxSceneAPI.CreateBroadphaseTypeAttr("MBP")
    physxSceneAPI.CreateSolverTypeAttr("TGS")

    # Setting up Ground Plane
    PhysicsSchemaTools.addGroundPlane(
        stage, "/World/groundPlane", "Z", 15, Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.7)
    )

    # Define the root Xform (transformable object)
    rootxform = UsdGeom.Xform.Define(stage, "/World")

    rigid_body_x = [1.0, -1.0]

    ang_vel = [[0.0, 0.0, 90.0], [00.0, 00.0, 90.0]]

    sm_quat_rot = UQ.AngVec(90, [0, 0, 1], unit="deg")

    quat_body = [Gf.Quatf(*sm_quat_rot.vec.tolist()), Gf.Quatf(1.0)]

    rigid_api = []
    rigidBodyXform = []
    for i in range(2):
        # Create an xformable
        rigidBodyPath = f"/World/rigidBody{i}"

        rigidBodyXform.append(UsdGeom.Xform.Define(stage, rigidBodyPath))
        rigidBodyXform[-1].AddTranslateOp().Set(Gf.Vec3f(rigid_body_x[i], 0.0, 0.75))
        rigidBodyXform[-1].AddOrientOp().Set(quat_body[i])
        rigidBodyPrim = rigidBodyXform[-1].GetPrim()
        rigid_api.append(UsdPhysics.RigidBodyAPI.Apply(rigidBodyPrim))

        # Adding a Cube
        size = 0.5
        cube_pos = Gf.Vec3f(0.0, 0.0, 0.0)
        cube_quat = Gf.Quatf(1.0)
        cube_scale = Gf.Vec3f(1.0, 1.0, 1.0)

        cube_path = rigidBodyPath + "/Cube"
        cubeGeom = UsdGeom.Cube.Define(stage, cube_path)
        cubeGeom.CreateSizeAttr(size)
        cubeGeom.CreateExtentAttr([Gf.Vec3f(-size / 2), Gf.Vec3f(size / 2)])
        cubeGeom.AddTranslateOp().Set(cube_pos)
        cubeGeom.AddOrientOp().Set(cube_quat)
        cubeGeom.AddScaleOp().Set(cube_scale)

        cubePrim = stage.GetPrimAtPath(cube_path)

        # Attach Rigid Body and Collision Preset
        # rigid_api.CreateRigidBodyEnabledAttr(True)
        UsdPhysics.CollisionAPI.Apply(cubePrim)

        linVel = Gf.Vec3f(0.0, 0.0, 0.0)
        angVel = Gf.Vec3f(*ang_vel[i])
        rigid_api[-1].CreateVelocityAttr(linVel)
        rigid_api[-1].CreateAngularVelocityAttr(angVel)
        timeline = omni.timeline.get_timeline_interface()

        # sim_view = physx.create_simulation_view("torch")
        # rigid_body_view = sim_view.create_rigid_body_view("/World/rigidBody0")

    first_run = True
    while simulation_app.is_running():

        if first_run:
            first_run = False
        # get_physx_simulation_interface().simulate(1/60.0, 40 * 1/60 + 1 * 1/60)
        # # Fetch results will wait for simulation to finish and push simulation results to the output
        # get_physx_simulation_interface().fetch_results()
        transform_matrix = rigidBodyXform[0].ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        transform = Gf.Transform(transform_matrix)
        H_w0 = np.array(transform.GetMatrix()).T
        transform_matrix = rigidBodyXform[1].ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        transform = Gf.Transform(transform_matrix)
        H_w1 = np.array(transform.GetMatrix()).T

        pos = transform.GetTranslation()
        quat = transform.GetRotation().GetQuat()
        time = timeline.get_current_time()
        sm_quat_rot = UQ(quat.GetReal(), np.asarray(quat.GetImaginary()))

        print(f"{'=':=^20}")
        print(
            f"Time: {np.round(time,3)}, RPY [deg]: {sm_quat_rot.rpy().round(4)}, ang_vec [deg]: {sm_quat_rot.angvec('deg')}"
        )
        print(
            f"RPY [deg/s]: {(sm_quat_rot.rpy('deg')/(time + 1e-4)).round(4)}, rate ang_vec [deg/s]: {(sm_quat_rot.angvec('deg')[0] * sm_quat_rot.angvec('deg')[1])/(time + 1e-4)}"
        )
        print(
            f"RPY [rad/s]: {(sm_quat_rot.rpy('rad')/(time + 1e-4)).round(4)}, rate ang_vec [rad/s]: {(sm_quat_rot.angvec('rad')[0] * sm_quat_rot.angvec('rad')[1])/(time + 1e-4)}"
        )
        print(f"{'=':=^20}")

        asyncio.ensure_future(rigid_body_abstraction(H_w0, H_w1))

        # rigidBodyCircleVelocities = Gf.Vec3f(5*np.sin(0.1*time), 5*np.cos(0.1*time), 0.0)
        rigidBodyCircleVelocities = Gf.Vec3f(
            -1.0, 0.0, 0.0
        )  # Gf.Vec3f(0.0, 0.0, 0.5*np.cos(0.1*time))
        rigid_api[1].GetVelocityAttr().Set(rigidBodyCircleVelocities)

        # rigidBodyCircleVelocities = Gf.Vec3f(0.0, 0.0, 0.5*np.sin(0.1*time))
        rigidBodyCircleVelocities = Gf.Vec3f(
            0.0, 0.0, 0.0
        )  # Gf.Vec3f(0.0, 0.0, 0.5*np.cos(0.1*time))
        rigid_api[0].GetVelocityAttr().Set(rigidBodyCircleVelocities)

        simulation_app.update()


if __name__ == "__main__":
    main()
    simulation_app.close()
