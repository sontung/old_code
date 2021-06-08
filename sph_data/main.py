import sys

import numpy as np

sys.path.append("../libraries/libgeom")

import pysplishsplash as sph
import os
import pathlib
import open3d as o3d
from tqdm import tqdm
from airbag_sampler import create_airbag_pointclouds
from lib import surface_reconstruct_marching_cube, remove_inside_mesh
from lib import surface_reconstruct_marching_cube_with_vis, mesh_filtering
import glob
import meshio
import time


def build_shape(radius=(40, 30, 35), model_file='airbag.obj'):
    large_radius_torus, small_radius_torus, radius_sphere = radius
    pcd = create_airbag_pointclouds(large_radius_torus, small_radius_torus, radius_sphere)

    vertices, faces = surface_reconstruct_marching_cube(point_cloud=pcd, cube_size=0.75, isovalue=0.7)
    remove_list, face_status, kept_vertices, kept_faces = remove_inside_mesh(vertices, faces)

    kept_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(kept_vertices),
                                          o3d.utility.Vector3iVector(kept_faces))
    lineset = o3d.geometry.LineSet.create_from_triangle_mesh(kept_mesh)
    lineset.paint_uniform_color([0, 0, 0])

    o3d.visualization.draw_geometries([lineset])
    o3d.io.write_triangle_mesh(model_file, kept_mesh)


def sph_simulation():
    print("running sph simulation")
    base = sph.Exec.SimulatorBase()
    output_dir = os.path.abspath("../data_heavy/sph_solutions")

    root_path = pathlib.Path(__file__).parent.absolute()
    scene_file_path = os.path.join(root_path, 'EmitterModel_option_1.json')
    base.init(sceneFile=scene_file_path, outputDir=output_dir, useGui=False)
    base.setValueFloat(base.STOP_AT, 25.0)  # Important to have the dot to denote a float
    base.setValueFloat(base.DATA_EXPORT_FPS, 5)
    base.run()


def vtk_to_mesh(vtk_folder='../data_heavy/sph_solutions/vtk/', if_vis=False):
    vtk_files = glob.glob(vtk_folder + '*.vtk')
    vtk_files = sorted(vtk_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    list_mesh = []
    for file in tqdm(vtk_files, desc="Extracting meshes from vtk files"):
        data = meshio.read(file)
        point_cloud = data.points
        pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(point_cloud))

        v, f = surface_reconstruct_marching_cube(pcd, cube_size=0.15, isovalue=0.14, verbose=False)
        nv, nf, _, _ = mesh_filtering(v, f, if_vis=False, verbose=True)

        mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(nv),
                                         triangles=o3d.utility.Vector3iVector(nf))

        o3d.io.write_triangle_mesh("mc_solutions/%s.obj" % file.split("/")[-1].split(".")[0], mesh)

        list_mesh.append(mesh)

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


def txt_to_mesh(txt_folder='../data_heavy/sph_solutions/new_state/', if_vis=False):
    txt_files = glob.glob(txt_folder + '*.txt')
    txt_files = sorted(txt_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    list_mesh = []
    for file in tqdm(txt_files, desc="Extracting meshes from vtk files"):
        data = o3d.io.read_point_cloud(file, format='xyzrgb')
        point_cloud = np.asarray(data.points)
        pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(point_cloud))

        nv, nf = surface_reconstruct_marching_cube(pcd, cube_size=0.15, isovalue=0.14, verbose=False)
        # nv, nf, _, _ = mesh_filtering(v, f, if_vis=False, verbose=True)

        mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(nv),
                                         triangles=o3d.utility.Vector3iVector(nf))
        o3d.io.write_triangle_mesh("mc_solutions/%s.obj" % file.split("/")[-1].split(".")[0], mesh)

        list_mesh.append(mesh)

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

            while ang-prev_ang <= 1000:
                ang += 10
                vis.poll_events()
                vis.update_renderer()
            prev_ang = ang
            vis.capture_screen_image("saved/im%d.png" % ind)
            ind += 1
    return list_mesh


if __name__ == "__main__":
    start = time.time()
    os.makedirs("mc_solutions", exist_ok=True)
    os.makedirs("mc_solutions_smoothed", exist_ok=True)
    # build_shape()
    # sph_simulation()
    # output_mesh = vtk_to_mesh(if_vis=False)
    txt_to_mesh(if_vis=True)
    print("SPH simulation done in %f" % (time.time()-start))
