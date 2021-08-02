import argparse
import glob
import math
import os
import pickle
import sys
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as rot_mat_compute
from tqdm import tqdm

from anomaly_detetion import neutralize_head_rot
from anomaly_detetion import look_for_abnormals_based_on_ear_sizes_tight
from custom_rigid_cpd import RigidRegistration
from laplacian_fairing_1d import laplacian_fairing
from rec_utils import b_spline_smooth, normalize, draw_text_to_image, partition_by_none, partition_by_not_none
from solve_airbag import compute_ab_pose, compute_ab_frames, compute_head_ab_areas_image_space
from custom_model import new_model

DEBUG_MODE=True


def check_translation_bound(head_traj, ab_transx, ab_transy, special_interval):
    """
    scale the translation to reach the bound
    """
    print(f"scale to match airbag pose = ({ab_transx}, {ab_transy})")
    os.makedirs("../data_heavy/area_compute/", exist_ok=True)
    pcd = o3d.io.read_triangle_mesh("../data/max-planck.obj")
    ab = o3d.io.read_triangle_mesh("../data/max-planck.obj")

    start_ab, _ = compute_ab_frames()
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    ab.translate([0, -ab_transx, -ab_transy])
    head_x_pos = []
    head_y_pos = []
    original_pos = pcd.get_center()
    pcd.translate(np.array([0, 0, 0]), relative=False)
    for counter in range(len(head_traj)):
        pcd.translate(head_traj[counter % len(head_traj)])
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        head_x_pos.append(pcd.get_center()[1])
        head_y_pos.append(pcd.get_center()[2])

    if DEBUG_MODE:
        print("before y", np.min(head_y_pos), np.max(head_y_pos), -ab_transy)
        print("before x", np.min(head_x_pos), np.max(head_x_pos), -ab_transx)

    head_x_pos_old = head_x_pos[:]
    head_y_pos_old = head_y_pos[:]
    if len(head_traj) != special_interval[1]-1:
        print("scaling head vertical trajectory:")
        if special_interval is None:
            print(" normal scaling")
            scale_x = abs((abs(ab_transx))/np.max(head_x_pos))
            head_x_pos = [du * scale_x for du in head_x_pos]

            scale_y = abs((abs(ab_transy)) / np.min(head_y_pos))
            head_y_pos = [du * scale_y for du in head_y_pos]

        else:
            print(" scaling specifically using an interval")
            start, end = map(int, special_interval)
            mi = np.min(head_x_pos[start:end])
            head_x_pos = [du - ab_transx-mi for du in head_x_pos]

            mi = np.min(head_y_pos[start:end])
            ma = np.max(head_y_pos[start:end])
            head_y_pos = [10*(du-mi)/(ma-mi)-ab_transy+5 for du in head_y_pos]

    # recompute trajectory
    prev_pos = original_pos[1:]
    trajectories = []
    for idx in range(len(head_x_pos)):
        mean = np.array([head_x_pos[idx], head_y_pos[idx]])
        if prev_pos is not None:
            trans = np.zeros((3, 1))
            move = mean - prev_pos
            trans[2] = move[1]
            trans[1] = move[0]
            trajectories.append(trans)
        prev_pos = mean

    # re-simulate
    new_head_x_pos = []
    new_head_y_pos = []
    pcd.translate(original_pos-pcd.get_center())
    if DEBUG_MODE:
        print("at", pcd.get_center(), original_pos)
        for counter in range(len(trajectories)):
            pcd.translate(trajectories[counter % len(head_traj)])
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            new_head_x_pos.append(pcd.get_center()[1])
            new_head_y_pos.append(pcd.get_center()[2])

        print("after y", np.min(new_head_y_pos), np.max(new_head_y_pos), -ab_transy)
        print("after x", np.min(new_head_x_pos), np.max(new_head_x_pos), -ab_transx)

        plt.subplot(211)
        plt.plot(head_x_pos_old)
        plt.plot(new_head_x_pos)
        plt.plot([-ab_transx-5]*len(new_head_x_pos))
        plt.plot([-ab_transx+5]*len(new_head_x_pos))

        plt.legend(["ori", "scaled", "bound"])

        plt.subplot(212)
        plt.plot(head_y_pos_old)
        plt.plot(new_head_y_pos)
        plt.plot([-ab_transy]*len(new_head_y_pos))
        plt.legend(["ori", "scaled", "bound"])
        plt.savefig("trans_bound.png")
        plt.close()

    vis.destroy_window()
    return trajectories


def compute_translation(ab_transx, ab_transy):
    """
    translation of the head
    """
    img = cv2.imread("../data_heavy/frames/1-1.png")
    x_dim, y_dim, _ = img.shape
    sys.stdin = open("../data_heavy/frames/info.txt")
    lines = [du[:-1] for du in sys.stdin.readlines()]
    all_pixel_dir = "../data_heavy/frames_ear_coord_only"

    sys.stdin = open("../data_heavy/frame2ab.txt")
    lines2 = [du[:-1] for du in sys.stdin.readlines()]
    frame2ab = {du.split(" ")[0]: du for du in lines2}

    trajectories = []
    x_traj = []
    y_traj = []
    first_disappear = None
    start_counting = False
    for idx in lines:
        _, _, head_pixels = frame2ab[f"1-{idx}.png"].split(" ")[:3]
        head_pixels = int(head_pixels)

        if head_pixels <= 2 and first_disappear is None:
            first_disappear = [idx, idx]
            start_counting = True
        elif start_counting and head_pixels <= 2:
            first_disappear[1] = idx
        elif start_counting and head_pixels > 2:
            start_counting = False

        with open("%s/1-%s.png" % (all_pixel_dir, idx), "rb") as fp:
            right_pixels_all = pickle.load(fp)
        if len(right_pixels_all) == 0:
            print(f" did not detect ear in 1-{idx}.png")
            mean = [None, None]
        else:
            mean = np.mean(np.array(right_pixels_all), axis=0)

            mean[0] = mean[0] / x_dim
            mean[1] = mean[1] / y_dim

        x_traj.append(mean[0])
        y_traj.append(mean[1])

    ranges = partition_by_not_none(x_traj)
    longest = -1
    for start, end in ranges:
        if end-start > longest:
            first_disappear = [start, end]
            longest = end-start
    print("detecting head into airbag between", first_disappear)

    ab_transx_new = ab_transx
    ab_transy_new = ab_transy

    # ab_transx_new = ab_transx-y_traj[0]
    # ab_transy_new = ab_transy-x_traj[0]
    #
    ab_transx_new = ab_transx_new/y_dim
    ab_transy_new = ab_transy_new/x_dim

    x_traj = look_for_abnormals_based_on_ear_sizes_tight(x_traj)
    y_traj = look_for_abnormals_based_on_ear_sizes_tight(y_traj)

    x_traj1 = laplacian_fairing(x_traj)
    y_traj1 = laplacian_fairing(y_traj)

    start, end = first_disappear
    shift_x = ab_transy_new - x_traj1[start]
    shift_y = ab_transx_new - y_traj1[start]
    x_traj = [dux+shift_x for dux in x_traj if dux is not None]
    y_traj = [duy+shift_y for duy in y_traj if duy is not None]

    for idx in range(start, end):
        x_traj[idx] = ab_transy_new
        y_traj[idx] = ab_transx_new

    x_traj = laplacian_fairing(x_traj, "lapx.png", None)
    y_traj = laplacian_fairing(y_traj, "lapy.png", None)

    ab_transy_new = ab_transy_new*100
    ab_transx_new = ab_transx_new*100
    x_traj *= 100
    y_traj *= 100

    # main
    prev_pos = np.array([0, 0])
    for idx in tqdm(range(len(x_traj)), desc="Computing head x-y translation"):
        mean = np.array([x_traj[idx], y_traj[idx]])
        if prev_pos is not None:
            trans = np.zeros((3, 1))
            move = mean - prev_pos
            trans[2] = -move[1]
            trans[1] = -move[0]
            trajectories.append(trans)
        prev_pos = mean

    # sys.exit()

    # if int(first_disappear[1]) - int(first_disappear[0]) <= 5:
    #     first_disappear = None
    # trajectories = check_translation_bound(trajectories, ab_transy_new, ab_transx_new, first_disappear)

    return trajectories, x_traj, y_traj, ab_transx_new, ab_transy_new


if __name__ == '__main__':
    ab_mesh_dir = "../sph_data/mc_solutions_smoothed"

    pcd = new_model()
    ab_scale, ab_transx, ab_transy, ab_rot, ab_area, head_area = compute_ab_pose()

    # scale the AB to match the scale head/ab in image space
    global_scale_ab_list = []
    for ab_dir in glob.glob(f"{ab_mesh_dir}/*"):
        ab = o3d.io.read_triangle_mesh(ab_dir)
        scale1 = pcd.get_surface_area() / ab.get_surface_area()
        global_scale_ab_list.append(math.sqrt(scale1 / ab_scale))
    global_scale_ab = np.mean(global_scale_ab_list)
    trajectory, ne_trans_x_traj, ne_trans_y_traj, ab_transx2, ab_transy2 = compute_translation(ab_transx, ab_transy)
    start_ab, _ = compute_ab_frames()
    ab_counter = 0
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.add_geometry(pcd)
    vis.get_view_control().set_zoom(2.5)
    arr = []
    vis.get_view_control().rotate(-500, 0)
    ab_centersx = []
    ab_centersy = []

    head_centersx = []
    head_centersy = []
    print("airbag at", -ab_transy2, -ab_transx2)
    pcd.translate(trajectory[0], relative=False)
    trajectory = trajectory[1:]
    for counter in tqdm(range(len(trajectory)), desc="Completing simulation"):
        ab_added = False
        pcd.translate(trajectory[counter % len(trajectory)])
        vis.update_geometry(pcd)

        if counter >= start_ab-1:
            ab = o3d.io.read_triangle_mesh(f"{ab_mesh_dir}/new_particles_%d.obj" % ab_counter)
            ab.compute_vertex_normals()
            ab.scale(1000, ab.get_center())
            ab.translate([0, -ab_transy2, -ab_transx2])
            ab.rotate(rot_mat_compute.from_euler("y", 90, degrees=True).as_matrix())
            ab.rotate(rot_mat_compute.from_euler("x", -90+ab_rot, degrees=True).as_matrix())
            ab_added = True
            vis.add_geometry(ab, reset_bounding_box=False)
            ab_counter += 1
            ab_centersx.append(ab.get_center()[1])
            ab_centersy.append(ab.get_center()[2])
        else:
            ab_centersx.append(None)
            ab_centersy.append(None)
        head_centersx.append(pcd.get_center()[1])
        head_centersy.append(pcd.get_center()[2])

        vis.update_renderer()
        vis.poll_events()
        if ab_added:
            vis.remove_geometry(ab, reset_bounding_box=False)
    vis.destroy_window()

    plt.subplot(211)
    plt.plot(head_centersx)
    plt.plot(ab_centersx)
    plt.plot([42,57],[head_centersx[42], head_centersx[57]], "bo")
    plt.title("x")
    plt.subplot(212)
    plt.plot(head_centersy)
    plt.plot(ab_centersy)
    plt.plot([42,57],[head_centersy[42], head_centersy[57]], "bo")

    plt.title("y")

    plt.savefig("test_trans.png")
    plt.close()