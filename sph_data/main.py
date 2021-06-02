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
        f = mesh_filtering(v, f, if_vis=False, verbose=True)

        mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(v),
                                         triangles=o3d.utility.Vector3iVector(f))
        o3d.io.write_triangle_mesh("mc_solutions/%s.obj" % file.split("/")[-1].split(".")[0], mesh)

        mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(v),
                                         triangles=o3d.utility.Vector3iVector(f))
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


if __name__ == "__main__":
    start = time.time()
    build_shape()
    sph_simulation()
    output_mesh = vtk_to_mesh(if_vis=False)
    print("SPH simulation done in %f" % (time.time()-start))
