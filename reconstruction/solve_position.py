import pickle
import sys
import numpy as np
import open3d as o3d
import cv2
import kmeans1d
import time
from scipy.spatial.transform import Rotation as rot_mat_compute


def compute_translation():
    """
    translation of the head
    """
    sys.stdin = open("../data_heavy/frames/info.txt")
    lines = [du[:-1] for du in sys.stdin.readlines()]
    all_pixel_dir = "../data_heavy/frames_ear_coord_only"
    refined_pixel_dir = "../data_heavy/refined_pixels"
    images_dir = "../data_heavy/frames_ear_only_nonblack_bg"

    trajectories = []
    prev_pos = None
    for idx in lines:
        with open("%s/1-%s.png" % (all_pixel_dir, idx), "rb") as fp:
            right_pixels_all = pickle.load(fp)
        mean = np.mean(np.array(right_pixels_all), axis=0)
        if prev_pos is not None:
            trans = np.zeros((3, 1))
            move = mean - prev_pos
            trans[2] = -move[1]
            trans[1] = move[0]
            trajectories.append(trans)
        prev_pos = mean
    new_list = []
    for i in reversed(trajectories):
        new_list.append(i*-1)
    trajectories.extend(new_list)
    return trajectories


def compute_rotation():
    sys.stdin = open("../data_heavy/frames/info.txt")
    lines = [du[:-1] for du in sys.stdin.readlines()]
    all_pixel_dir = "../data_heavy/frames_ear_coord_only"
    refined_pixel_dir = "../data_heavy/refined_pixels"
    images_dir = "../data_heavy/frames_ear_only_nonblack_bg"

    trajectories = []
    prev_pos = None
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    for idx in lines:
        with open("%s/1-%s.png" % (refined_pixel_dir, idx), "rb") as fp:
            right_pixels_edges = pickle.load(fp)
        img = cv2.imread("%s/1-%s.png" % (images_dir, idx))
        # img = np.zeros_like(img)
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
        max_line_gap = 20  # maximum gap in pixels between connectable line segments
        line_image = np.copy(img) * 0  # creating a blank to draw lines on

        edges = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        _, edges = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY)

        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

        angles = []
        angles_true = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
                rot_deg = np.rad2deg(np.arctan2(y2-y1, x2-x1))
                angles.append(abs(rot_deg))
                angles_true.append(rot_deg)
        clusters, centroids = kmeans1d.cluster(angles, 2)
        rot_deg_overall = np.mean([angles_true[i] for i in range(len(clusters)) if clusters[i] == 0])
        print(sorted(angles_true), rot_deg_overall)

        if prev_pos is not None:
            move = rot_deg_overall - prev_pos
            # trajectories.append(move)
            trajectories.append(rot_deg_overall)

        prev_pos = rot_deg_overall

        # new_img = cv2.addWeighted(img, 0.8, line_image, 1, 0)
        # new_img = np.vstack([new_img, line_image])
        # new_img = cv2.resize(new_img, (new_img.shape[1]//4, new_img.shape[0]//4))
        # cv2.imshow("test", new_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # break
    new_list = []
    print(trajectories)
    for i in reversed(trajectories):
        new_list.append(i*-1)
    trajectories.extend(new_list)
    print(trajectories)
    return trajectories


def visualize():
    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    # source_raw = o3d.io.read_triangle_mesh("../data/max-planck.obj")
    #
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(source_raw)
    # for i in range(len(trajectories)):
    #     source_raw.translate(trajectories[i])
    #     vis.update_geometry(source_raw)
    #     vis.poll_events()
    #     vis.update_renderer()
    # vis.destroy_window()

    global pcd, trajectory, counter, rotated_trajectory, degg, parameters
    pcd = o3d.io.read_triangle_mesh("../data/max-planck.obj")
    trajectory = compute_translation()
    rotated_trajectory = compute_rotation()
    assert len(rotated_trajectory) == len(trajectory)
    counter = 0

    degg = 10
    parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2021-05-11-15-20-43.json")

    def rotate_view(vis):
        global pcd, trajectory, counter, rotated_trajectory, degg, parameters
        pcd.translate(trajectory[counter % len(trajectory)])

        rot_mat = rot_mat_compute.from_euler('x', rotated_trajectory[counter % len(trajectory)],
                                             degrees=True).as_matrix()

        pcd.rotate(rot_mat, pcd.get_center())
        vis.update_geometry(pcd)
        counter += 1
        if counter > len(trajectory):
            counter = 0
        time.sleep(0.5)

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(parameters)

        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd],
                                                              rotate_view)


if __name__ == '__main__':
    # compute_rotation()
    visualize()