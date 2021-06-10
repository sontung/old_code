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
from scipy import interpolate
from matplotlib import pyplot as plt
from tqdm import tqdm


def compute_ab_frames():
    sys.stdin = open("../data_heavy/frames/info.txt")
    lines = [du[:-1] for du in sys.stdin.readlines()]
    sys.stdin = open("../data_heavy/frame2ab.txt")
    lines2 = [du[:-1] for du in sys.stdin.readlines()]
    frame2ab = {u: int(v) for u, v in [du.split(" ") for du in lines2]}
    traj = []
    for frn in lines:
        akey = "1-%s.png" % frn
        traj.append(frame2ab[akey])
    for idx, num in enumerate(traj):
        if np.all([traj[idx+du] > 1000 for du in range(5)]):
            return idx, len(traj)-idx-1
    raise RuntimeError


def b_spline_smooth(_trajectory, vis=False, return_params=False):
    """
    b spline smoothing for missing values (denoted None)
    Args:
        _trajectory:

    Returns:
    """
    control_points = []
    control_points_time = []
    not_there = []
    for idx, computation in enumerate(_trajectory):
        if computation is not None:
            control_points.append(computation)
            control_points_time.append(idx)
        else:
            not_there.append(idx)
    tck = interpolate.splrep(control_points_time, control_points, k=3)
    values = [interpolate.splev(du, tck) for du in np.linspace(0, len(_trajectory), len(_trajectory))]
    if vis:
        plt.plot(control_points_time, control_points, "ob")
        plt.plot(not_there, [interpolate.splev(du, tck) for du in not_there], "or")

        plt.plot(np.linspace(0, len(_trajectory), 1000),
                 [interpolate.splev(du, tck) for du in np.linspace(0, len(_trajectory), 1000)], "y")
        plt.xlabel("time")
        plt.ylabel("position")
        plt.legend(["available points", "missing points", "interpolated curve"], prop={'size': 15})
        plt.savefig('/home/sontung/Downloads/Figure_1.png', dpi=300)
    if return_params:
        return tck
    return values


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


def compute_rotation(reverse_for_vis=False):
    sys.stdin = open("../data_heavy/frames/info.txt")
    lines = [du[:-1] for du in sys.stdin.readlines()]
    refined_pixel_dir = "../data_heavy/refined_pixels"
    images_dir = "../data_heavy/frames_ear_only_nonblack_bg"
    trajectories = []
    prev_pos = None
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    all_angles = []

    for idx in tqdm(lines, desc="Computing head rotation"):
        with open("%s/1-%s.png" % (refined_pixel_dir, idx), "rb") as fp:
            right_pixels_edges = pickle.load(fp)
        img = cv2.imread("%s/1-%s.png" % (images_dir, idx))
        for x, y in right_pixels_edges:

            boundary = False
            for u, v in neighbors:
                if img[x + u, y + v, 0] == 128 and img[x + u, y + v, 1] == 128 and img[x + u, y + v, 2] == 255:
                    boundary = True
                    break
            if not boundary:
                img[x, y] = 255

        kernel = np.ones((5, 5), np.uint8)
        temp = cv2.erode(img, kernel)
        temp = cv2.dilate(temp, kernel)
        temp = cv2.subtract(img, temp)

        # lines
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 50  # minimum number of pixels making up a line
        max_line_gap = 30  # maximum gap in pixels between connectable line segments
        line_image = np.copy(img) * 0  # creating a blank to draw lines on

        edges = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        _, edges = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY)

        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
        if lines is None:
            all_angles.append(None)
            continue

        angles = []
        angles_true = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
                rot_deg = np.rad2deg(np.arctan2(y2-y1, x2-x1))
                angles.append(abs(rot_deg))
                angles_true.append(rot_deg)
        #         print(x2, x1, y2, y1)
        #         cv2.imshow("test", line_image)
        #         cv2.waitKey()
        #         cv2.destroyAllWindows()
        #
        # cv2.imshow("test", img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        clusters, centroids = kmeans1d.cluster(angles, 2)
        rot_deg_overall = np.mean([angles_true[i] for i in range(len(clusters)) if clusters[i] == 0])
        all_angles.append(rot_deg_overall)
    #     cv2.imwrite("%s/1-%s.png" % (line_images_dir, idx), line_image)
    #     print(rot_deg_overall, angles, "%s/1-%s.png" % (images_dir, idx))
    # sys.exit()

    all_angles = b_spline_smooth(all_angles)
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
    os.makedirs("../data_heavy/saved/", exist_ok=True)
    global pcd, trajectory, counter, rotated_trajectory, parameters, parameters2
    global ab_counter
    pcd = o3d.io.read_triangle_mesh("../data/max-planck.obj")
    pcd.compute_vertex_normals()

    trajectory = compute_translation()
    rotated_trajectory = compute_rotation()

    start_ab, _ = compute_ab_frames()
    counter = 0
    ab_counter = 0
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_view_control().set_zoom(1.5)

    # vis.get_view_control().rotate(-500, 0)
    # vis.get_view_control().rotate(500, 0)
    # vis.get_view_control().rotate(-200, 0)

    # global du11
    # du11 = 1.5
    # def cb(avis):
    #     avis.get_view_control().rotate(-500, 0)
    #     avis.capture_screen_image("../data_heavy/saved/v1.png", do_render=True)
    #     avis.get_view_control().rotate(500, 0)
    #
    #     avis.get_view_control().rotate(-200, 0)
    #     avis.capture_screen_image("../data_heavy/saved/v2.png", do_render=True)
    #     avis.get_view_control().rotate(200, 0)
    #     return False
    #
    # vis.register_animation_callback(cb)
    # vis.run()  # user changes the view and press "q" to terminate
    # vis.destroy_window()
    # sys.exit()

    def rotate_view(avis):
        global pcd, trajectory, counter, rotated_trajectory, parameters, parameters2
        global ab_counter
        ab_added = False
        pcd.translate(trajectory[counter % len(trajectory)]/5)
        rot_mat = rot_mat_compute.from_euler('x', rotated_trajectory[counter % len(trajectory)],
                                             degrees=True).as_matrix()
        pcd.rotate(rot_mat, pcd.get_center())

        avis.update_geometry(pcd)
        counter += 1
        if counter > len(trajectory)-1:
            counter = 0
            sys.exit()
        if counter >= start_ab+1:
            ab_counter += 1
            ab = o3d.io.read_triangle_mesh("../sph_data/mc_solutions_smoothed/new_particles_%d.obj" % ab_counter)
            ab.compute_vertex_normals()
            ab.scale(180.0, ab.get_center())
            ab.compute_vertex_normals()
            ab.translate([0, 0, -250])
            ab.rotate(rot_mat_compute.from_euler("y", 90, degrees=True).as_matrix())
            ab.rotate(rot_mat_compute.from_euler("x", -10, degrees=True).as_matrix())
            ab_added = True
            avis.add_geometry(ab)
        avis.get_view_control().rotate(-500, 0)
        avis.capture_screen_image("../data_heavy/saved/v1-%s.png" % counter, do_render=True)

        avis.get_view_control().rotate(500, 0)
        avis.get_view_control().rotate(-200, 0)
        avis.capture_screen_image("../data_heavy/saved/v2-%s.png" % counter, do_render=True)
        avis.get_view_control().rotate(200, 0)
        if ab_added:
            avis.remove_geometry(ab)

        return False

    vis.register_animation_callback(rotate_view)
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    # test()
    visualize()


    # from scipy import interpolate
    #
    # trajectory = compute_translation(reverse_for_vis=False)
    # trajectory = np.hstack(trajectory)
    # trajectory = np.delete(trajectory, 0, 0)
    # trajectory = np.transpose(trajectory)
    # print(trajectory[:, 0].shape)
    #
    # x2y = {du1: trajectory[du1, 0] for du1 in range(trajectory[:, 0].shape[0])}
    #
    # x_points = list(range(trajectory[:, 0].shape[0]))
    # y_points = trajectory[:, 0]
    # tck = interpolate.splrep(x_points, y_points, k=3)
    #
    # values = [interpolate.splev(du, tck) for du in np.linspace(0, 31, 1000)]
    # plt.plot(x_points, y_points, "ob")
    # plt.plot(np.linspace(0, 31, 1000), values)
    # plt.show()

