from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from pxr import UsdPhysics, PhysxSchema, Gf, PhysicsSchemaTools, UsdGeom
import omni

from omni.physx import get_physx_simulation_interface

def main():

    stage = omni.usd.get_context().get_stage()

    # Setting up Physics Scene
    gravity = 9.8
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

    # Adding a Cube
    path = "/World/Cube"
    cubeGeom = UsdGeom.Cube.Define(stage, path)
    cubePrim = stage.GetPrimAtPath(path)
    size = 0.5
    offset = Gf.Vec3f(0.5,0.2,1.0)
    cubeGeom.CreateSizeAttr(size)
    cubeGeom.AddTranslateOp().Set(offset)

    # Attach Rigid Body and Collision Preset
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(cubePrim)
    rigid_api.CreateRigidBodyEnabledAttr(True)
    UsdPhysics.CollisionAPI.Apply(cubePrim)
    
    # get_physx_simulation_interface().attach_stage(stage_id)

    while simulation_app.is_running():
        get_physx_simulation_interface().simulate(1/60.0, 40 * 1/60 + 1 * 1/60)

        # Fetch results will wait for simulation to finish and push simulation results to the output
        get_physx_simulation_interface().fetch_results()
        simulation_app.update()

if __name__ == "__main__":
    main()
    simulation_app.close()
    
