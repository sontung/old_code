import open3d as o3d
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as rot_mat_compute


def new_model(debugging=False):

    texture = cv2.imread("../data/model/textures/Head_albedo.jpg")
    texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)

    pcd_old = o3d.io.read_triangle_mesh("../data/max-planck.obj")
    pcd_old.compute_vertex_normals()
    pcd_old.translate([0, 0, 0])

    pcd = o3d.io.read_triangle_mesh("../data/model/model.obj")
    pcd.compute_vertex_normals()

    triangle_uvs = np.asarray(pcd.triangle_uvs)
    triangle_uvs[:, 1] = 1 - triangle_uvs[:, 1]

    pcd.textures = [o3d.geometry.Image(texture)]

    # scale new_model to old_model
    area1 = pcd.get_surface_area()
    area_scale = 980
    pcd.scale(area_scale, pcd.get_center())
    print(f"Area of head model: {pcd_old.get_surface_area()}\nArea of new head model: {area1}, "
          f"new head model after scale: {pcd.get_surface_area()}")

    # rotation new model
    rot_mat = rot_mat_compute.from_euler('y', -180,
                                         degrees=True).as_matrix()
    pcd.rotate(rot_mat, pcd.get_center())

    # sync center of 2 model
    center1 = pcd_old.get_center()
    center2 = pcd.get_center()

    diff = center1 - center2
    pcd.translate(diff)
    center3 = pcd.get_center()

    if debugging:

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_view_control().set_zoom(1.5)

        vis.add_geometry(pcd)
        vis.add_geometry(pcd_old)

        t = 0
        while t < 200:
            vis.poll_events()
            vis.update_renderer()
            t += 1

    return pcd
