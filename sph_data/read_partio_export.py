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


def write_pointcloud(pcd, save_file):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    with open(save_file, 'w') as file:
        for i in range(len(points)):
            s = f"{points[i][0]} {points[i][1]} {points[i][2]} {colors[i][0]} {colors[i][1]} {colors[i][2]}\n"
            file.write(s)
        file.close()
    return


def read_vtk_file(directory='../data_heavy/sph_solutions/vtk/', save_dir='/media/hblab/01D5F2DD5173DEA0/Anhntn1/libgeom/pointcloud'):
    vtk_files = glob.glob(directory + '*.vtk')
    vtk_files = sorted(vtk_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    for file in vtk_files:
        print(file)
        data = meshio.read(file)
        pcd = array2pointcloud(data.points)

        name_file = os.path.basename(file).replace('vtk', 'txt')
        write_pointcloud(pcd, os.path.join(save_dir, name_file))
    return


def read_vtk_file_with_vis(directory='../data_heavy/sph_solutions/vtk/', save_image='image_gif'):

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    ctr = vis.get_view_control()
    cnt_angle = 0.0

    ctr.rotate(cnt_angle, 0)
    i = 0

    vtk_files = glob.glob(directory + '*.vtk')
    vtk_files = sorted(vtk_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    for file in vtk_files:
        # print(file)
        data = meshio.read(file)
        pcd = array2pointcloud(data.points)
        vis.clear_geometries()
        vis.add_geometry(pcd)
        # cnt_angle += 5
        # ctr.rotate(cnt_angle, 0)
        vis.poll_events()
        vis.update_renderer()

        vis.capture_screen_image("image_gif/gif_%d.png" % i)
        i += 1

    # for file in vtk_files[::-1]:
    #     data = meshio.read(file)
    #     pcd = array2pointcloud(data.points)
    #     vis.clear_geometries()
    #     vis.add_geometry(pcd)
    #     # cnt_angle += 5
    #     # ctr.rotate(cnt_angle, 0)
    #     vis.poll_events()
    #     vis.update_renderer()
    #
    #     vis.capture_screen_image("image_gif/gif_%d.jpg" % i)
    #     i += 1

    return


if __name__ == "__main__":
    read_vtk_file_with_vis()
# data = meshio.read("../data_heavy/sph_solutions/vtk/ParticleData_Fluid_26.vtk")
# print(data.points)

