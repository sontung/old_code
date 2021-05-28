import sys

import open3d.visualization
import pysplishsplash as sph
import os
import pathlib
import numpy as np
import open3d as o3d
from airbag_sampler import create_airbag_pointclouds
from libraries.libgeom.lib import surface_reconstruct_marching_cube, remove_inside_mesh
from libraries.libgeom.utils import read_obj_file_texture_coords
import glob
import meshio


def build_shape(radius, pointclouds_file="../data_heavy/airbag.pcd", model_file='airbag.obj'):
    large_radius_torus, small_radius_torus, radius_sphere = radius
    pcd = create_airbag_pointclouds(large_radius_torus, small_radius_torus, radius_sphere)
    o3d.io.write_point_cloud(pointclouds_file, pcd)

    vertices, faces = surface_reconstruct_marching_cube(point_cloud=pcd)
    remove_list, face_status, vertices, keep_faces = remove_inside_mesh(vertices, faces)

    kept_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                          o3d.utility.Vector3iVector(keep_faces))
    o3d.io.write_triangle_mesh(model_file, kept_mesh)


def sph_simulation():
    base = sph.Exec.SimulatorBase()
    output_dir = os.path.abspath("../data_heavy/sph_solutions")

    root_path = pathlib.Path(__file__).parent.absolute()
    scene_file_path = os.path.join(root_path, 'EmitterModel_option_1.json')
    base.init(sceneFile=scene_file_path, outputDir=output_dir)
    base.setValueFloat(base.STOP_AT, 25)  # Important to have the dot to denote a float
    # base.setValueBool(base.VTK_EXPORT, True)
    base.setValueFloat(base.DATA_EXPORT_FPS, 5)
    gui = sph.GUI.Simulator_GUI_imgui(base)
    base.setGui(gui)
    base.run()


def vtk_to_mesh(vtk_folder='../data_heavy/sph_solutions/vtk/'):
    vtk_files = glob.glob(vtk_folder + '*.vtk')
    vtk_files = sorted(vtk_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    list_mesh = []
    for file in vtk_files:
        # print(file)
        data = meshio.read(file)
        point_cloud = data.points
        pcd = open3d.geometry.PointCloud(points=open3d.utility.Vector3dVector(point_cloud))

        v, f = surface_reconstruct_marching_cube(pcd)
        mesh = open3d.geometry.TriangleMesh(vertices=open3d.utility.Vector3dVector(v),
                                            triangles=open3d.utility.Vector3iVector(f))
        list_mesh.append(mesh)

    return list_mesh


if __name__ == "__main__":
    # sample()
    radius = (40, 30, 35)
    build_shape(radius)
    sph_simulation()
    output_mesh = vtk_to_mesh()

