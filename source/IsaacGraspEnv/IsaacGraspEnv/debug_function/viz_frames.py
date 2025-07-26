from typing import Optional
import torch
import open3d as o3d
from spatialmath.base import q2r
from modern_robotics import RpToTrans


def o3d_viz_body_key_points_obj_frames(
    body_pos: torch.Tensor,
    key_points: torch.Tensor,
    object_path: str,
    object_pos: torch.Tensor,
    object_quat: torch.Tensor,
    body_quat: Optional[torch.Tensor] = None,
):

    torch2numpy = lambda v: v.cpu().numpy()
    mesh = o3d.io.read_triangle_mesh(object_path)
    mesh.transform(RpToTrans(q2r(torch2numpy(object_quat)), torch2numpy(object_pos)))

    mesh_frames = []

    for m, pos in enumerate(body_pos):
        mesh_frames.append(
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=torch2numpy(pos))
        )
        if body_quat:
            mesh_frames[-1].rotate(q2r(torch2numpy(body_quat[m])))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(torch2numpy(key_points))

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    for frame in mesh_frames:
        vis.add_geometry(frame)
    vis.add_geometry(mesh)

    # Change point size via render options
    render_option = vis.get_render_option()
    render_option.point_size = 5.0  # Set to desired size (default is often 1.0)

    vis.run()
    vis.destroy_window()
