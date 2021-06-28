import os
import math
import glob
import pickle
import sys
import numpy as np
import open3d as o3d
import cv2
import time
from scipy.spatial.transform import Rotation as rot_mat_compute
from rec_utils import b_spline_smooth, normalize, draw_text_to_image, refine_path_computation
from tqdm import tqdm
from solve_airbag import compute_ab_pose, compute_ab_frames
from pycpd import RigidRegistration
import matplotlib.pyplot as plt
import time
import argparse

DEBUG_MODE = True


def compute_translation(debugging=DEBUG_MODE):
    """
    translation of the head
    """
    sys.stdin = open("../data_heavy/frames/info.txt")
    lines = [du[:-1] for du in sys.stdin.readlines()]
    all_pixel_dir = "../data_heavy/frames_ear_coord_only"

    trajectories = []
    x_traj = []
    y_traj = []
    prev_pos = None
    for idx in lines:
        with open("%s/1-%s.png" % (all_pixel_dir, idx), "rb") as fp:
            right_pixels_all = pickle.load(fp)
        if len(right_pixels_all) == 0:
            print(f"did not detect ear in 1-{idx}.png")
            mean = [None, None]
        else:
            mean = np.mean(np.array(right_pixels_all), axis=0)
        x_traj.append(mean[0])
        y_traj.append(mean[1])

    # b spline interpolation
    if debugging:
        b_spline_smooth(x_traj, vis=True, name=f"trans_x_ori.png")
        b_spline_smooth(y_traj, vis=True, name=f"trans_y_ori.png")

        x_traj = refine_path_computation(x_traj)
        y_traj = refine_path_computation(y_traj)
        x_traj = b_spline_smooth(x_traj, vis=True, name=f"trans_x_refined.png")
        y_traj = b_spline_smooth(y_traj, vis=True, name=f"trans_y_refined.png")
    else:
        x_traj = refine_path_computation(x_traj)
        y_traj = refine_path_computation(y_traj)
        x_traj = b_spline_smooth(x_traj)
        y_traj = b_spline_smooth(y_traj)

    for idx in tqdm(range(len(x_traj)), desc="Computing head x-y translation"):
        mean = np.array([x_traj[idx], y_traj[idx]])
        if prev_pos is not None:
            trans = np.zeros((3, 1))
            move = mean - prev_pos
            trans[2] = -move[1]
            trans[1] = move[0]
            trajectories.append(trans)
        prev_pos = mean
    return trajectories


def compute_rotation_accurate(debugging=DEBUG_MODE):
    sys.stdin = open("../data_heavy/frames/info.txt")
    lines = [du[:-1] for du in sys.stdin.readlines()]
    images_dir = "../data_heavy/line_images"
    trajectories = []
    prev_pos = None
    all_angles = []
    os.makedirs("../data_heavy/rigid_head_rotation", exist_ok=True)
    for idx in tqdm(lines[:], desc="Computing head x-y rotation using rigid CPD"):
        img = cv2.imread(f"{images_dir}/1-{idx}.png")
        if img is None or np.sum(img) == 0:
            all_angles.append(None)
            continue
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        nonzero_indices = np.nonzero(image)
        target_matrix = np.zeros((nonzero_indices[0].shape[0], 2))
        for i in range(nonzero_indices[0].shape[0]):
            target_matrix[i] = [nonzero_indices[0][i], nonzero_indices[1][i]]
        source_matrix = np.loadtxt('../data/ear.txt')

        source_matrix_norm = normalize(source_matrix, target_matrix)
        reg = RigidRegistration(**{'X': target_matrix, 'Y': source_matrix_norm}, max_iterations=45)
        y_data_norm, (_, rot_mat, _) = reg.register()
        rot_angle = np.rad2deg(np.arctan2(rot_mat[1, 0], rot_mat[0, 0]))
        all_angles.append(rot_angle)

        fig = plt.figure()
        fig.add_axes([0, 0, 1, 1])
        ax = fig.axes[0]
        ax.scatter(target_matrix[:, 0],  target_matrix[:, 1], color='red', label='Target')
        ax.scatter(y_data_norm[:, 0],  y_data_norm[:, 1], color='yellow', label='Source')
        ax.legend(loc='upper left', fontsize='x-large')
        plt.savefig(f"../data_heavy/rigid_head_rotation/1-{idx}.png")
        plt.close(fig)

    if debugging:
        stamp = time.time()
        b_spline_smooth(all_angles, vis=True, name=f"rot_ori_{stamp}.png")
        all_angles = refine_path_computation(all_angles)
        all_angles = b_spline_smooth(all_angles, vis=True, name=f"rot_smooth_{stamp}.png")
    else:
        all_angles = refine_path_computation(all_angles)
        all_angles = b_spline_smooth(all_angles)

    for rot_deg_overall in all_angles:
        if prev_pos is not None:
            move = rot_deg_overall - prev_pos
            trajectories.append(-move)
        prev_pos = rot_deg_overall

    if debugging:
        pcd = o3d.io.read_triangle_mesh("../data/max-planck.obj")
        pcd.compute_vertex_normals()
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)
        vis.add_geometry(pcd)
        vis.get_view_control().set_zoom(1.5)
        vis.get_view_control().rotate(-500, 0)

        for idx, rot in enumerate(trajectories):
            rot_mat = rot_mat_compute.from_euler('x', rot, degrees=True).as_matrix()
            pcd.rotate(rot_mat, pcd.get_center())
            time.sleep(1)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

    return trajectories, all_angles


def compute_rotation(reverse_for_vis=False, view=1):
    sys.stdin = open("../data_heavy/frames/info.txt")
    trajectories = []
    prev_pos = None

    sys.stdin = open("../data_heavy/frames/info.txt")
    lines = [du[:-1] for du in sys.stdin.readlines()]
    sys.stdin = open("../data_heavy/frame2ab.txt")
    lines2 = [du[:-1] for du in sys.stdin.readlines()]
    frame2ab = {du.split(" ")[0]: du for du in lines2}

    # check view exist
    all_key = list(frame2ab.keys())
    views = [int(key.split('-')[0]) for key in all_key]
    if view not in views:
        print(f"View {view} does not exist")
        return None

    rot_all = []
    for frn in tqdm(lines, desc=f"Computing head y-z rotation by view {view}"):
        akey = "%s-%s.png" % (view, frn)
        _, ab_area, head_area, _, _, head_rot, _ = frame2ab[akey].split(" ")
        head_area = float(head_area)
        try:
            if head_area > 1000:
                head_rot = float(head_rot)
                rot_all.append(head_rot)
            else:
                rot_all.append(None)
        except ValueError:
            rot_all.append(None)

    all_angles = b_spline_smooth(rot_all)

    for rot_deg_overall in all_angles:
        if prev_pos is not None:
            move = rot_deg_overall - prev_pos
            trajectories.append(-move)
        prev_pos = rot_deg_overall
    if reverse_for_vis:
        new_list = []
        for i in reversed(trajectories):
            new_list.append(i*-1)
        trajectories.extend(new_list)
    return trajectories


def compute_head_ab_areas(sim_first=False):
    if sim_first:
        ab_mesh_dir = "../sph_data/mc_solutions_smoothed"
        os.makedirs("../data_heavy/area_compute/", exist_ok=True)
        pcd = o3d.io.read_triangle_mesh("../data/max-planck.obj")
        ab_scale, ab_transx, ab_transy, ab_rot, ab_area, head_area = compute_ab_pose()

        global_scale_ab_list = []
        for ab_dir in glob.glob(f"{ab_mesh_dir}/*"):
            ab = o3d.io.read_triangle_mesh(ab_dir)
            scale1 = pcd.get_surface_area() / ab.get_surface_area()
            global_scale_ab_list.append(math.sqrt(scale1 / ab_scale))
        global_scale_ab = np.mean(global_scale_ab_list)

        trajectory = compute_translation()
        rotated_trajectory_z = compute_rotation(view=2)
        rotated_trajectory, ne_rot_traj = compute_rotation_accurate()
        if rotated_trajectory is None:
            rotated_trajectory = compute_rotation()

        if rotated_trajectory_z is not None:
            assert len(trajectory) == len(rotated_trajectory) == len(rotated_trajectory_z)
        else:
            assert len(trajectory) == len(rotated_trajectory)

        start_ab, _ = compute_ab_frames()
        mesh_files = glob.glob("%s/*" % ab_mesh_dir)
        print(f"Trajectory has {len(trajectory)} points, airbag starts at {start_ab} with {len(mesh_files)} meshes")
        ab_counter = 0
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(pcd)
        vis.get_view_control().set_zoom(1.5)
        pcd.compute_vertex_normals()
        arr = []
        for counter in tqdm(range(len(trajectory)), desc="Prior sim to compute view areas"):
            ab_added = False
            pcd.translate(trajectory[counter % len(trajectory)]/5)
            rot_mat = rot_mat_compute.from_euler('x', rotated_trajectory[counter],
                                                 degrees=True).as_matrix()
            pcd.rotate(rot_mat, pcd.get_center())

            if rotated_trajectory_z is not None:
                rot_mat_z = rot_mat_compute.from_euler('z', rotated_trajectory_z[counter],
                                                       degrees=True).as_matrix()
                pcd.rotate(rot_mat_z, pcd.get_center())

            vis.update_geometry(pcd)

            vis.get_view_control().rotate(-500, 0)
            vis.capture_screen_image("../data_heavy/area_compute/head-%s.png" % counter, do_render=True)
            if counter >= start_ab:
                ab = o3d.io.read_triangle_mesh(f"{ab_mesh_dir}/new_particles_{ab_counter}.obj")
                arr.append(f"{ab_mesh_dir}/new_particles_{ab_counter}.obj")
                ab.compute_vertex_normals()
                ab.scale(global_scale_ab, ab.get_center())
                ab.translate([0, -ab_transx, -ab_transy])
                ab.rotate(rot_mat_compute.from_euler("y", 90, degrees=True).as_matrix())
                ab.rotate(rot_mat_compute.from_euler("x", -90+ab_rot, degrees=True).as_matrix())
                ab_added = True
                vis.remove_geometry(pcd, reset_bounding_box=False)
                vis.add_geometry(ab, reset_bounding_box=False)
                vis.capture_screen_image("../data_heavy/area_compute/ab-%s.png" % counter, do_render=True)
                vis.add_geometry(pcd, reset_bounding_box=False)
                ab_counter += 1

            if ab_added:
                vis.remove_geometry(ab, reset_bounding_box=False)
            vis.get_view_control().rotate(500, 0)
            vis.poll_events()
            vis.update_renderer()
        vis.destroy_window()

    image_names = glob.glob("../data_heavy/area_compute/head*")
    arr = []
    for name in image_names:
        im = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        arr.append(np.sum(im != 255))
    head_area = np.mean(arr)

    image_names = glob.glob("../data_heavy/area_compute/ab*")
    arr = []
    for name in image_names:
        im = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        arr.append(np.sum(im != 255))
    ab_area = np.mean(arr)
    if sim_first:
        return head_area, ab_area, trajectory, rotated_trajectory, rotated_trajectory_z, ne_rot_traj
    return head_area, ab_area


def visualize(debug_mode=DEBUG_MODE):
    ab_mesh_dir = "../sph_data/mc_solutions_smoothed"
    os.makedirs("../data_heavy/saved/", exist_ok=True)
    pcd = o3d.io.read_triangle_mesh("../data/max-planck.obj")
    pcd.compute_vertex_normals()
    ab_scale, ab_transx, ab_transy, ab_rot, ab_area, head_area = compute_ab_pose()
    sim_head_area, sim_ab_area, trajectory, rotated_trajectory, rotated_trajectory_z, ne_rot_traj = compute_head_ab_areas(sim_first=True)
    translation_scale = math.sqrt(sim_head_area/head_area)
    scale2 = math.sqrt(ab_area/head_area*sim_head_area/sim_ab_area)
    global_scale_ab_list = []
    for ab_dir in glob.glob(f"{ab_mesh_dir}/*"):
        ab = o3d.io.read_triangle_mesh(ab_dir)
        scale1 = pcd.get_surface_area() / ab.get_surface_area()
        global_scale_ab_list.append(math.sqrt(scale1 / ab_scale))
    global_scale_ab = np.mean(global_scale_ab_list)*scale2
    print(f"Airbag pose: translation=({ab_transx}, {ab_transy}), rotation={ab_rot}, scale={global_scale_ab}, translation scale = {translation_scale}")

    start_ab, _ = compute_ab_frames()
    ab_counter = 0
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.get_view_control().set_zoom(1.5)

    for counter in tqdm(range(len(trajectory)), desc="Completing simulation"):
        ab_added = False
        pcd.translate(trajectory[counter % len(trajectory)]/5)
        rot_mat = rot_mat_compute.from_euler('x', rotated_trajectory[counter % len(trajectory)],
                                             degrees=True).as_matrix()
        pcd.rotate(rot_mat, pcd.get_center())

        if rotated_trajectory_z is not None:
            rot_mat_z = rot_mat_compute.from_euler('z', rotated_trajectory_z[counter % len(trajectory)],
                                                   degrees=True).as_matrix()
            pcd.rotate(rot_mat_z, pcd.get_center())

        vis.update_geometry(pcd)

        if counter >= start_ab+1:
            ab_counter += 1
            ab = o3d.io.read_triangle_mesh(f"{ab_mesh_dir}/new_particles_%d.obj" % ab_counter)
            ab.compute_vertex_normals()
            ab.scale(global_scale_ab, ab.get_center())
            ab.translate([0, -ab_transx*translation_scale, -ab_transy*translation_scale])
            ab.rotate(rot_mat_compute.from_euler("y", 90, degrees=True).as_matrix())
            ab.rotate(rot_mat_compute.from_euler("x", -90+ab_rot, degrees=True).as_matrix())
            ab_added = True
            vis.add_geometry(ab, reset_bounding_box=False)
        vis.get_view_control().rotate(-500, 0)
        vis.capture_screen_image("../data_heavy/saved/v1-%s.png" % counter, do_render=True)

        vis.get_view_control().rotate(500, 0)
        vis.get_view_control().rotate(-200, 0)
        vis.capture_screen_image("../data_heavy/saved/v2-%s.png" % counter, do_render=True)
        vis.get_view_control().rotate(200, 0)
        if ab_added:
            vis.remove_geometry(ab, reset_bounding_box=False)

    vis.destroy_window()
    if debug_mode:
        sys.stdin = open("../data_heavy/frames/info.txt")
        lines = [du[:-1] for du in sys.stdin.readlines()]
        ear_dir = "../data_heavy/frames_ear_only"
        rigid_dir = "../data_heavy/rigid_head_rotation"
        images_dir = "../data_heavy/line_images"
        print(lines)
        lines = lines[1:]
        print(len(lines), len(trajectory))
        assert len(lines) == len(trajectory)
        for counter, ind in enumerate(lines):
            line_img = cv2.imread("%s/1-%s.png" % (images_dir, ind))
            ear_img = cv2.imread("%s/1-%s.png" % (ear_dir, ind))
            rigid_img = cv2.imread("%s/1-%s.png" % (rigid_dir, ind))
            arr = np.nonzero(ear_img)
            if len(arr[0]) > 0:
                ear_img = ear_img[np.min(arr[0]):np.max(arr[0]), np.min(arr[1]): np.max(arr[1])]
                cv2.imwrite(f"test/ear-{ind}.png", ear_img)
                cv2.imwrite(f"test/line-{ind}.png", line_img)
                cv2.imwrite(f"test/rigid-{ind}.png", rigid_img)
            im1 = cv2.imread("../data_heavy/saved/v1-%s.png" % counter)
            im1 = draw_text_to_image(im1, "rot=%.3f" % (ne_rot_traj[counter]))
            im1 = cv2.resize(im1, (im1.shape[1]//2, im1.shape[0]//2))
            cv2.imwrite(f"test2/res-{ind}.png", im1)


if __name__ == '__main__':
    # test()
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', '-d', )
    visualize(False)
