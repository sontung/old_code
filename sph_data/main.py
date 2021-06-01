import sys

import numpy as np

sys.path.append("../libraries/libgeom")

import pysplishsplash as sph
import os
import pathlib
import open3d as o3d
from tqdm import tqdm
from airbag_sampler import create_airbag_pointclouds
from lib import surface_reconstruct_marching_cube, remove_inside_mesh, surface_reconstruct_marching_cube_with_vis, loop_subdivision
import glob
import meshio
from collections import Counter


def build_shape(radius=(40, 30, 35), model_file='airbag.obj'):
    large_radius_torus, small_radius_torus, radius_sphere = radius
    pcd = create_airbag_pointclouds(large_radius_torus, small_radius_torus, radius_sphere)

    vertices, faces = surface_reconstruct_marching_cube(point_cloud=pcd)
    remove_list, face_status, vertices, keep_faces = remove_inside_mesh(vertices, faces)

    kept_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                          o3d.utility.Vector3iVector(keep_faces))
    o3d.io.write_triangle_mesh(model_file, kept_mesh)


def sph_simulation():
    print("running sph simulation")
    base = sph.Exec.SimulatorBase()
    output_dir = os.path.abspath("../data_heavy/sph_solutions")

    root_path = pathlib.Path(__file__).parent.absolute()
    scene_file_path = os.path.join(root_path, 'EmitterModel_option_1.json')
    base.init(sceneFile=scene_file_path, outputDir=output_dir)
    base.setValueFloat(base.STOP_AT, 25.0)  # Important to have the dot to denote a float
    base.setValueFloat(base.DATA_EXPORT_FPS, 5)
    gui = sph.GUI.Simulator_GUI_imgui(base)
    base.setGui(gui)
    base.run()


def remove_noise_of_mesh(faces):
    face_status = {(x, y, z): -1 for x, y, z in faces}

    cnt = -1
    while len(faces) > 0:
        check_vertices = list(faces[0])
        status = True

        cnt += 1
        flag = len(faces)
        while status:
            for face in faces:
                if face[0] in check_vertices or face[1] in check_vertices or face[2] in check_vertices:
                    check_vertices.extend(face)
                    face_status[(face[0], face[1], face[2])] = cnt
            faces = [k for k, v in face_status.items() if v == -1]
            if len(faces) < flag:
                flag = len(faces)
            else:
                status = False

    values = Counter(list(face_status.values()))
    max_key = max(values, key=values.get)
    big_face = [k for k, v in face_status.items() if v == max_key]
    return np.array(big_face)


def vtk_to_mesh(vtk_folder='../data_heavy/sph_solutions/vtk/', if_vis=False):
    vtk_files = glob.glob(vtk_folder + '*.vtk')
    vtk_files = sorted(vtk_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    list_mesh = []
    for file in tqdm(vtk_files, desc="Extracting meshes from vtk files"):
        data = meshio.read(file)
        point_cloud = data.points
        pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(point_cloud))

        v, f = surface_reconstruct_marching_cube(pcd, cube_size=0.15, isovalue=0.14, verbose=False)
        v, f = loop_subdivision(v, f)

        f = remove_noise_of_mesh(f)

        mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(v),
                                         triangles=o3d.utility.Vector3iVector(f))
        o3d.visualization.draw_geometries([mesh])

        o3d.io.write_triangle_mesh("mc_solutions/%s.obj" % file.split("/")[-1].split(".")[0], mesh)

        mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(v),
                                         triangles=o3d.utility.Vector3iVector(f))
        list_mesh.append(mesh)
        # surface_reconstruct_marching_cube_with_vis(pcd)

    if if_vis:
        vis = o3d.visualization.Visualizer()

        vis.create_window()
        ctr = vis.get_view_control()
        prev_ang = 0
        ang = 0
        ind = 0
        for mesh in list_mesh:
            vis.clear_geometries()
            mesh.compute_vertex_normals()
            original_mesh_wf = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
            original_mesh_wf.paint_uniform_color([0, 0, 0])
            original_mesh_wf.translate([3, 0, 0])
            vis.add_geometry(original_mesh_wf)
            vis.add_geometry(mesh)
            ctr.rotate(250, 0.0)

            while ang-prev_ang <= 20:
                ang += 10
                vis.poll_events()
                vis.update_renderer()
            prev_ang = ang
            vis.capture_screen_image("saved/im%d.png" % ind)
            ind += 1
    return list_mesh


def draw_line(vertices, faces):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()

    for i, face in enumerate(faces):
        print(i)
        lines = [[0, 1], [1, 2], [2, 0]]
        colors = [[0, 0, 1] for i in range(len(lines))]
        triangle_points = np.array([vertices[face[0]], vertices[face[1]], vertices[face[2]]])
        line_pcd = o3d.geometry.LineSet()
        line_pcd.lines = o3d.utility.Vector2iVector(lines)
        line_pcd.colors = o3d.utility.Vector3dVector(colors)
        line_pcd.points = o3d.utility.Vector3dVector(triangle_points)

        vis.add_geometry(line_pcd)
        ctr.rotate(10, 0.0)
        vis.poll_events()
        vis.update_renderer()

    return


if __name__ == "__main__":
    # sample()
    # build_shape()
    # sph_simulation()
    output_mesh = vtk_to_mesh(if_vis=True)

