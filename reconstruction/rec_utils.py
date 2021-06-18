import sys
import csv
import cv2
import os
import json
import open3d
import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt


def b_spline_smooth(_trajectory, vis=False, name="test2.png", return_params=False):
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
        plt.savefig(f'{name}', dpi=300)
        plt.close()
    if return_params:
        return tck
    return values


def compute_zncc(x, y, x2, y2, f, g, f_, g_, window_size, using_global_mean=True):
    """
    zncc score for a pair of image patches (fast)
    """
    f = f[x-window_size: x+window_size+1, y-window_size: y+window_size+1]
    g = g[x2-window_size: x2+window_size+1, y2-window_size: y2+window_size+1]
    if not using_global_mean:
        f_ = [np.mean(f[:, :, c]) for c in range(3)]
        g_ = [np.mean(g[:, :, c]) for c in range(3)]
    du1 = np.multiply(f-f_, g-g_)
    du2 = np.multiply(f-f_, f-f_)
    du3 = np.multiply(g-g_, g-g_)
    s2 = np.sum(du1) / (np.sqrt(np.sum(du2) * np.sum(du3)) + 0.00001)

    # from PIL import Image
    # Image.fromarray(np.hstack([f, g])).save("debugs/%d%d%d%d.png" % (x, y, x2, y2))

    return s2, f, g


def dump_into_tracks_osfm(corr_dir, im_names, mats, csv_dir):
    """
    convert correspondences into opensfm format
    """
    pairs = read_correspondence_from_dump(corr_dir)
    out = {im: [] for im in im_names}
    w, h, _ = mats[0].shape
    size = max(w, h)
    for track_id, (x1, y1, x2, y2) in enumerate(pairs):
        (x1, y1, x2, y2) = map(int, (x1, y1, x2, y2))
        r1, g1, b1 = mats[0][x1, y1]
        r2, g2, b2 = mats[1][x2, y2]

        # normalize as opensfm format
        x1 = (x1 + 0.5 - w / 2.0) / size
        y1 = (y1 + 0.5 - h / 2.0) / size
        x2 = (x2 + 0.5 - w / 2.0) / size
        y2 = (y2 + 0.5 - h / 2.0) / size

        out[im_names[0]].append((im_names[0], track_id, track_id, y1, x1, 1, r1, g1, b1, -1, -1))
        out[im_names[1]].append((im_names[1], track_id, track_id, y2, x2, 1, r2, g2, b2, -1, -1))
    sys.stdout = open(csv_dir, "w")
    print("OPENSFM_TRACKS_VERSION_v2")
    for k in out:
        for row in out[k]:
            row = map(str, row)
            print("\t".join(row))


def read_correspondence_from_dump(txt_dir="data/corr-3.txt"):
    sys.stdin = open(txt_dir, "r")
    lines = sys.stdin.readlines()
    pairs = [tuple(map(float, line[:-1].split(" "))) for line in lines]
    return pairs


def complement_point_cloud():
    """
    complement the tracks.csv to remove bad matches (e.g. outside of the seg mask)
    :return:
    """

    data = {}
    with open('data_heavy/sfm_data/tracks.csv', newline='') as csvfile:
        lines = csv.reader(csvfile, delimiter='\t', quotechar='|')
        next(lines)

        for row in lines:
            im_name, trackid, _, x, y, _, r, g, b = row[:9]
            if trackid not in data:
                data[trackid] = [(im_name, trackid, x, y, r, g, b)]
            else:
                data[trackid].append((im_name, trackid, x, y, r, g, b))

    im_dict = {f: cv2.imread(os.path.join("data_heavy/sfm_data/images/", f))
               for f in os.listdir("data_heavy/sfm_data/images/")
               if os.path.isfile(os.path.join("data_heavy/sfm_data/images/", f))}
    mask_dict = {f: cv2.imread(os.path.join("data_heavy/sfm_data/masks/", f))
                 for f in os.listdir("data_heavy/sfm_data/masks/")
                 if os.path.isfile(os.path.join("data_heavy/sfm_data/masks/", f))}
    good_tracks = []
    for trackid in data:
        good_track = True
        for im_name, trackid, x, y, r, g, b in data[trackid]:
            im = im_dict[im_name]
            mask = mask_dict[im_name]
            h, w, _ = im.shape
            size = max(w, h)
            x, y = map(float, (x, y))
            x = int(x * size - 0.5 + w / 2.0)
            y = int(y * size - 0.5 + h / 2.0)

            if sum(mask[y, x]) == 0:
                good_track = False
                break
        if good_track:
            good_tracks.append(trackid)
    row_new = []
    with open('data_heavy/sfm_data/tracks.csv', newline='') as csvfile:
        lines = csv.reader(csvfile, delimiter='\t', quotechar='|')
        first_line = next(lines)
        for row in lines:
            if row[1] in good_tracks and row[0] in ["opencv_frame_0.png", "opencv_frame_1.png"]:
                row_new.append(row)
    sys.stdout = open('data_heavy/sfm_data/tracks2.csv', "w")
    print(first_line[0])
    for row in row_new:
        # row[5] = '1'
        # row[2] = "1794"
        print("\t".join(row))
    return "tracks2.csv"


def visualize_point_cloud(json_file="data_heavy/sfm_data/reconstruction.json"):
    """
    visualize the reconstructed point cloud
    :return:
    """
    with open(json_file) as a_file:
        data = json.load(a_file)[0]

    pc_out = open("data_heavy/point_cloud.txt", 'w')
    coord = []
    for k in data["points"]:
        xyz = data["points"][k]["coordinates"]
        rgb = data["points"][k]["color"]
        coord.append(xyz)

        print(xyz[0], xyz[1], xyz[2], rgb[0]/255, rgb[1]/255, rgb[2]/255, file=pc_out)
    ic(np.mean(np.array(coord)))
    pcd = open3d.io.read_point_cloud("data_heavy/point_cloud.txt", format='xyzrgb')

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(0.0, 0.0)
        return False

    open3d.visualization.draw_geometries_with_animation_callback([pcd], rotate_view)


def normalize(inp, ref):
    inp[:, 0] = (np.max(ref[:, 0]) - np.min(ref[:, 0]))*(inp[:, 0] - np.min(inp[:, 0]))/(np.max(inp[:, 0]) - np.min(inp[:, 0])) + np.min(ref[:, 0])
    inp[:, 1] = (np.max(ref[:, 1]) - np.min(ref[:, 1]))*(inp[:, 1] - np.min(inp[:, 1]))/(np.max(inp[:, 1]) - np.min(inp[:, 1])) + np.min(ref[:, 1])
    return inp

if __name__ == '__main__':
    dump_into_tracks_osfm()
    # visualize_point_cloud()
    # complement_point_cloud()
