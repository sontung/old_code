import os
import cv2
import sys
import rewrite_images
import open3d as o3d
import numpy as np
import pickle
import argparse
from scipy.spatial.transform import Rotation as rot_mat_compute

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', type=str, default=False, help='directory', required=True)

args = vars(parser.parse_args())
RESULTS_DIR = args['dir']
COMPUTATIONS_DIR = f"{RESULTS_DIR}/everything_you_need.pkl"

RENDER_MODE = 0
NEXT = False
PREV = False
MODIFIED_MODE = False


def key_cb(u):
    global RENDER_MODE
    if RENDER_MODE == 0:
        RENDER_MODE = 1
    elif RENDER_MODE == 1:
        RENDER_MODE = 0


def key_sw(u):
    global NEXT
    if not NEXT and RENDER_MODE == 0:
        NEXT = True


def key_sw2(u):
    global PREV
    if not PREV and RENDER_MODE == 0:
        PREV = True


def new_model(debugging=False):
    texture = cv2.imread("data/model/textures/Head_albedo.jpg")
    texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)

    pcd_old = o3d.io.read_triangle_mesh("data/max-planck.obj")
    pcd_old.compute_vertex_normals()

    pcd = o3d.io.read_triangle_mesh("data/model/model.obj")
    pcd.compute_vertex_normals()

    triangle_uvs = np.asarray(pcd.triangle_uvs)
    triangle_uvs[:, 1] = 1 - triangle_uvs[:, 1]

    pcd.textures = [o3d.geometry.Image(texture)]

    # scale new_model to old_model
    area_scale = 980
    pcd.scale(area_scale, pcd.get_center())

    # rotation new model
    rot_mat = rot_mat_compute.from_euler('y', -180, degrees=True).as_matrix()
    pcd.rotate(rot_mat, pcd.get_center())
    pcd.translate(pcd_old.get_center(), relative=False)

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


def modify_computed_data():
    with open(COMPUTATIONS_DIR, "rb") as pickle_file:
        all_computations = pickle.load(pickle_file)

    global_head_scale, global_ab_scale, trajectory, rotated_trajectory, \
    ab_transx, ab_transy, ab_rot, start_ab = all_computations

    global_ab_scale *= 1.5  # ab bigger
    for traj in trajectory:  # head goes further (x-axis)
        traj[2] -= 0.2

    return global_head_scale, global_ab_scale, trajectory, rotated_trajectory, \
    ab_transx, ab_transy, ab_rot, start_ab


def visualize():
    global NEXT, PREV

    ab_mesh_dir = f"{RESULTS_DIR}/mc_solutions_smoothed"
    os.makedirs("../data_heavy/saved/", exist_ok=True)
    pcd = new_model()
    pcd.translate([0, 0, 0], relative=False)
    pcd.compute_vertex_normals()
    if MODIFIED_MODE:
        new_data = modify_computed_data()
        rewrite_images.visualize(new_data, pcd)
        global_head_scale, global_ab_scale, trajectory, rotated_trajectory, \
        ab_transx, ab_transy, ab_rot, start_ab = new_data
    else:
        with open(COMPUTATIONS_DIR, "rb") as pickle_file:
            all_computations = pickle.load(pickle_file)

        global_head_scale, global_ab_scale, trajectory, rotated_trajectory, \
        ab_transx, ab_transy, ab_rot, start_ab = all_computations

    print(f"Airbag pose: translation=({ab_transx}, {ab_transy}), rotation="
          f"{ab_rot}, ab scale={global_ab_scale},"
          f" head scale = {global_head_scale}")
    ab_counter = 0
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(visible=True)
    vis.add_geometry(pcd)
    vis.get_view_control().set_zoom(1.5)
    vis.get_view_control().rotate(-500, 0)
    vis.register_key_callback(ord("A"), key_cb)
    vis.register_key_callback(ord("N"), key_sw)
    vis.register_key_callback(ord("P"), key_sw2)

    trans_actual_traj = []
    rot_actual_traj = []
    pcd.scale(global_head_scale, pcd.get_center())
    head_centers = []
    ab_centers = []
    ab_added_mode0 = False
    ab_added_mode1 = False
    prev_renders = []
    counter = 0
    pcd.translate(trajectory[0])

    while True:

        # manual mode
        if RENDER_MODE == 0:
            vis.poll_events()
            vis.update_renderer()

            if NEXT:
                counter += 1
                NEXT = False
                if ab_added_mode0:
                    vis.remove_geometry(ab, reset_bounding_box=False)
                    ab_added_mode0 = False
                vis.poll_events()
                pcd.translate(trajectory[counter % len(trajectory)])
                rot_mat = rot_mat_compute.from_euler('x', rotated_trajectory[counter % len(trajectory)],
                                                     degrees=True).as_matrix()
                pcd.rotate(rot_mat, pcd.get_center())
                trans_actual_traj.append(trajectory[counter % len(trajectory)])
                rot_actual_traj.append(rotated_trajectory[counter % len(trajectory)])

                if counter >= start_ab - 1:
                    ab = o3d.io.read_triangle_mesh(f"{ab_mesh_dir}/new_particles_%d.obj" % ab_counter)
                    ab.compute_vertex_normals()
                    ab.scale(global_ab_scale, ab.get_center())
                    ab.translate([0, -ab_transx, -ab_transy])
                    ab.rotate(rot_mat_compute.from_euler("y", 90, degrees=True).as_matrix())
                    ab.rotate(rot_mat_compute.from_euler("x", -90 + ab_rot, degrees=True).as_matrix())
                    vis.add_geometry(ab, reset_bounding_box=False)
                    ab_counter += 1
                    ab_added_mode0 = True

                vis.update_geometry(pcd)
                vis.update_renderer()
                prev_renders.append([-trajectory[counter % len(trajectory)],
                                     -rotated_trajectory[counter % len(trajectory)],
                                     0])

            elif PREV and len(prev_renders) > 0:
                counter -= 1
                PREV = False
                trans, rot, rot_z = prev_renders.pop()
                if ab_added_mode0:
                    vis.remove_geometry(ab, reset_bounding_box=False)
                    ab_added_mode0 = False
                vis.poll_events()
                pcd.translate(trans)
                rot_mat = rot_mat_compute.from_euler('x', rot, degrees=True).as_matrix()
                pcd.rotate(rot_mat, pcd.get_center())
                trans_actual_traj.append(trajectory[counter % len(trajectory)])
                rot_actual_traj.append(rotated_trajectory[counter % len(trajectory)])

                rot_mat_z = rot_mat_compute.from_euler('z', rot_z, degrees=True).as_matrix()
                pcd.rotate(rot_mat_z, pcd.get_center())

                if counter >= start_ab - 1:
                    ab_counter -= 2
                    ab = o3d.io.read_triangle_mesh(f"{ab_mesh_dir}/new_particles_%d.obj" % ab_counter)
                    ab.compute_vertex_normals()
                    ab.scale(global_ab_scale, ab.get_center())
                    ab.translate([0, -ab_transx, -ab_transy])
                    ab.rotate(rot_mat_compute.from_euler("y", 90, degrees=True).as_matrix())
                    ab.rotate(rot_mat_compute.from_euler("x", -90 + ab_rot, degrees=True).as_matrix())
                    vis.add_geometry(ab, reset_bounding_box=False)
                    ab_counter += 1
                    ab_added_mode0 = True

                vis.update_geometry(pcd)
                vis.update_renderer()

        # auto mode
        elif RENDER_MODE == 1:
            counter += 1
            vis.poll_events()
            if ab_added_mode1:
                ab_added_mode1 = False
                vis.remove_geometry(ab, reset_bounding_box=False)
            pcd.translate(trajectory[counter % len(trajectory)])
            rot_mat = rot_mat_compute.from_euler('x', rotated_trajectory[counter % len(trajectory)],
                                                 degrees=True).as_matrix()
            pcd.rotate(rot_mat, pcd.get_center())
            trans_actual_traj.append(trajectory[counter % len(trajectory)])
            rot_actual_traj.append(rotated_trajectory[counter % len(trajectory)])

            if counter >= start_ab-1:
                ab = o3d.io.read_triangle_mesh(f"{ab_mesh_dir}/new_particles_%d.obj" % ab_counter)
                ab.compute_vertex_normals()
                ab.scale(global_ab_scale, ab.get_center())
                ab.translate([0, -ab_transx, -ab_transy])
                ab.rotate(rot_mat_compute.from_euler("y", 90, degrees=True).as_matrix())
                ab.rotate(rot_mat_compute.from_euler("x", -90+ab_rot, degrees=True).as_matrix())
                ab_added_mode1 = True
                vis.add_geometry(ab, reset_bounding_box=False)
                ab_counter += 1
                ab_centers.append(ab.get_center())
            else:
                ab_centers.append(None)
            head_centers.append(pcd.get_center())

            vis.update_geometry(pcd)
            vis.update_renderer()
        if counter >= len(trajectory):
            vis.destroy_window()
            break


if __name__ == '__main__':
    while True:
        RENDER_MODE = 0
        visualize()
