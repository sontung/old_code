import os
import glob
import meshio
import numpy as np
import open3d


def array2pointcloud(array):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(array)
    pcd.paint_uniform_color(np.array([0, 0, 0]))
    return pcd


def read_vtk_file_with_vis(directory='../data_heavy/sph_solutions/vtk/', save_image='image_gif'):

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    ctr = vis.get_view_control()
    cnt_angle = 5.0
    i = 0

    vtk_files = glob.glob(directory + '*.vtk')
    vtk_files = sorted(vtk_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    for file in vtk_files:
        data = meshio.read(file)
        pcd = array2pointcloud(data.points)
        vis.clear_geometries()
        vis.add_geometry(pcd)
        ctr.rotate(cnt_angle, 0)
        # cnt_angle += 1
        vis.poll_events()
        vis.update_renderer()

        vis.capture_screen_image("image_gif/gif_%d.png" % i)
        i += 1

    for file in vtk_files[::-1]:
        data = meshio.read(file)
        pcd = array2pointcloud(data.points)
        vis.clear_geometries()
        vis.add_geometry(pcd)
        ctr.rotate(cnt_angle, 0)
        # cnt_angle += 1
        vis.poll_events()
        vis.update_renderer()

        vis.capture_screen_image("image_gif/gif_%d.jpg" % i)
        i += 1


    return

read_vtk_file_with_vis()
# data = meshio.read("../data_heavy/sph_solutions/vtk/ParticleData_Fluid_26.vtk")
# print(data.points)

