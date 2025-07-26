from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})


from pxr import UsdPhysics, PhysxSchema, Gf, PhysicsSchemaTools, UsdGeom, Usd
import omni
import numpy as np 
from spatialmath import UnitQuaternion as UQ

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
    PhysicsSchemaTools.addGroundPlane(stage, "/World/groundPlane", "Z", 15, Gf.Vec3f(0,0,0), Gf.Vec3f(0.7))
    
    # Define the root Xform (transformable object)
    rootxform = UsdGeom.Xform.Define(stage, "/World")
    
    # Create an xformable
    rigidBodyPath = "/World/rigidBody"
    
    
    rigidBodyXform = UsdGeom.Xform.Define(stage, rigidBodyPath)
    rigidBodyXform.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.75))
    rigidBodyXform.AddOrientOp().Set(Gf.Quatf(1.0))
    rigidBodyPrim = rigidBodyXform.GetPrim()
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(rigidBodyPrim)
    
    # Adding a Cube
    size = 0.5
    cube_pos = Gf.Vec3f(0.0,0.0,0.0)
    cube_quat = Gf.Quatf(1.0)
    cube_scale = Gf.Vec3f(1.0, 1.0, 1.0)
    
    cube_path = rigidBodyPath + "/Cube"
    cubeGeom = UsdGeom.Cube.Define(stage, cube_path)
    cubeGeom.CreateSizeAttr(size)
    cubeGeom.CreateExtentAttr([Gf.Vec3f(-size/2), Gf.Vec3f(size/2)])
    cubeGeom.AddTranslateOp().Set(cube_pos)
    cubeGeom.AddOrientOp().Set(cube_quat)
    cubeGeom.AddScaleOp().Set(cube_scale)
    
    cubePrim = stage.GetPrimAtPath(cube_path)
    
    # Attach Rigid Body and Collision Preset
    # rigid_api.CreateRigidBodyEnabledAttr(True)
    UsdPhysics.CollisionAPI.Apply(cubePrim)
    
    linVel = Gf.Vec3f(0.0, 0.0, 0.0)
    angVel = Gf.Vec3f(90.0, 0.0, 90.0)
    rigid_api.CreateVelocityAttr(linVel)
    rigid_api.CreateAngularVelocityAttr(angVel)
    timeline = omni.timeline.get_timeline_interface()
    
    while simulation_app.is_running():
        # get_physx_simulation_interface().simulate(1/60.0, 40 * 1/60 + 1 * 1/60)
        # # Fetch results will wait for simulation to finish and push simulation results to the output
        # get_physx_simulation_interface().fetch_results()
        transform_matrix = rigidBodyXform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        transform = Gf.Transform(transform_matrix)
        pos = transform.GetTranslation()
        quat = transform.GetRotation().GetQuat()
        
        sm_quat_rot = UQ(quat.GetReal(), np.asarray(quat.GetImaginary()))
        time = timeline.get_current_time()
        print(f"Time: {np.round(time,3)}, RPY [deg]: {sm_quat_rot.rpy().round(4)}, ang_vec [deg]: {sm_quat_rot.angvec('deg')}")
        print(f"RPY [deg/s]: {(sm_quat_rot.rpy('deg')/(time + 1e-4)).round(4)}, rate ang_vec [deg/s]: {(sm_quat_rot.angvec('deg')[0] * sm_quat_rot.angvec('deg')[1])/(time + 1e-4)}")
        print(f"==========================")
        simulation_app.update()

if __name__ == "__main__":
    main()
    simulation_app.close()
    