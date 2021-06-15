import os
import math
import glob
import pickle
import sys
import numpy as np
import open3d as o3d
import cv2
import kmeans1d
import time
from scipy.spatial.transform import Rotation as rot_mat_compute
from rec_utils import b_spline_smooth
from tqdm import tqdm
from solve_airbag import compute_ab_pose, compute_ab_frames


def compute_translation(reverse_for_vis=False):
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
            mean = [None, None]
        else:
            mean = np.mean(np.array(right_pixels_all), axis=0)
        x_traj.append(mean[0])
        y_traj.append(mean[1])

    # b spline interpolation
    x_traj, y_traj = map(b_spline_smooth, (x_traj, y_traj))
    for idx in tqdm(range(len(x_traj)), desc="Computing head position"):
        mean = np.array([x_traj[idx], y_traj[idx]])
        if prev_pos is not None:
            trans = np.zeros((3, 1))
            move = mean - prev_pos
            trans[2] = -move[1]
            trans[1] = move[0]
            trajectories.append(trans)
        prev_pos = mean
    if reverse_for_vis:
        new_list = []
        for i in reversed(trajectories):
            new_list.append(i*-1)
        trajectories.extend(new_list)
    return trajectories


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
    for frn in tqdm(lines, desc=f"Computing head rotation by view {view}"):
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

    # pcd = o3d.io.read_triangle_mesh("../data/max-planck.obj")
    # pcd.compute_vertex_normals()
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(visible=True)
    # vis.add_geometry(pcd)
    # vis.get_view_control().set_zoom(1.5)
    # vis.get_view_control().rotate(-500, 0)
    # move = 0
    # for idx, rot in enumerate(all_angles):
    #     if prev_pos is not None:
    #         move = rot - prev_pos
    #         trajectories.append(-move)
    #
    #     prev_pos = rot
    #     rot_mat = rot_mat_compute.from_euler('x', move, degrees=True).as_matrix()
    #     pcd.rotate(rot_mat, pcd.get_center())
    #     vis.update_geometry(pcd)
    #     vis.poll_events()
    #     vis.update_renderer()
    #     print(rot, "1-%s.png" % lines[idx])
    #     cv2.imshow("test", cv2.imread("../data_heavy/frames_seg_abh/%s" % ("1-%s.png" % lines[idx])))
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()
    # sys.exit()

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


def visualize():
    ab_mesh_dir = "../sph_data/mc_solutions_smoothed"
    os.makedirs("../data_heavy/saved/", exist_ok=True)
    global pcd, trajectory, counter, rotated_trajectory, rotated_trajectory_z
    global ab_counter
    pcd = o3d.io.read_triangle_mesh("../data/max-planck.obj")
    pcd.compute_vertex_normals()
    ab_scale, ab_transx, ab_transy, ab_rot = compute_ab_pose()

    global_scale_ab_list = []
    for ab_dir in glob.glob(f"{ab_mesh_dir}/*"):
        ab = o3d.io.read_triangle_mesh(ab_dir)
        scale1 = pcd.get_surface_area() / ab.get_surface_area()
        global_scale_ab_list.append(math.sqrt(scale1 / ab_scale))
    global_scale_ab = np.mean(global_scale_ab_list)

    print(f"Airbag pose: translation=({ab_transx}, {ab_transy}), rotation={ab_rot}, scale={ab_scale}")
    trajectory = compute_translation()
    rotated_trajectory = compute_rotation()
    rotated_trajectory_z = compute_rotation(view=2)

    start_ab, _ = compute_ab_frames()
    counter = 0
    ab_counter = 0
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.get_view_control().set_zoom(1.5)

    def rotate_view(avis):
        global pcd, trajectory, counter, rotated_trajectory, rotated_trajectory_z
        global ab_counter
        ab_added = False
        pcd.translate(trajectory[counter % len(trajectory)]/5)
        rot_mat = rot_mat_compute.from_euler('x', rotated_trajectory[counter % len(trajectory)],
                                             degrees=True).as_matrix()
        pcd.rotate(rot_mat, pcd.get_center())

        if rotated_trajectory_z is not None:
            rot_mat_z = rot_mat_compute.from_euler('z', rotated_trajectory_z[counter % len(trajectory)],
                                                   degrees=True).as_matrix()
            pcd.rotate(rot_mat_z, pcd.get_center())

        avis.update_geometry(pcd)
        counter += 1
        if counter > len(trajectory)-1:
            counter = 0
            sys.exit()
        if counter >= start_ab+1:
            ab_counter += 1
            ab = o3d.io.read_triangle_mesh(f"{ab_mesh_dir}/new_particles_%d.obj" % ab_counter)
            ab.compute_vertex_normals()
            ab.scale(global_scale_ab, ab.get_center())
            ab.translate([0, -ab_transy*2.5, -ab_transy*2.5])
            ab.rotate(rot_mat_compute.from_euler("y", 90, degrees=True).as_matrix())
            ab.rotate(rot_mat_compute.from_euler("x", -90+ab_rot, degrees=True).as_matrix())
            ab_added = True
            avis.add_geometry(ab, reset_bounding_box=False)
            # print(pcd.get_surface_area()/ab.get_surface_area())
        avis.get_view_control().rotate(-500, 0)
        avis.capture_screen_image("../data_heavy/saved/v1-%s.png" % counter, do_render=True)

        avis.get_view_control().rotate(500, 0)
        avis.get_view_control().rotate(-200, 0)
        avis.capture_screen_image("../data_heavy/saved/v2-%s.png" % counter, do_render=True)
        avis.get_view_control().rotate(200, 0)
        if ab_added:
            avis.remove_geometry(ab, reset_bounding_box=False)

        return False

    vis.register_animation_callback(rotate_view)
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    # test()
    visualize()

