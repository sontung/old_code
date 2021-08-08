import sys
import cv2
import json
import open3d
import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt


def grad_diff_compute(path, idx):
    grad = np.abs(np.diff(path))
    if grad[idx] <= 0.0001 or np.mean(grad[:idx]) <= 0.0001:
        return grad[idx]
    return grad[idx] / np.mean(grad[:idx])


def partition_by_none(path):
    ind = 0
    start = None
    ranges = []
    while ind < len(path):
        if path[ind] is not None and start is None:
            start = ind
        elif path[ind] is None and start is not None:
            end = ind
            ranges.append((start, end))
            start = None
        ind += 1
    if start is not None:
        ranges.append((start, ind))
    return ranges


def partition_by_not_none(path):
    ind = 0
    start = None
    ranges = []
    while ind < len(path):
        if path[ind] is None and start is None:
            start = ind
        elif path[ind] is not None and start is not None:
            end = ind
            ranges.append((start, end))
            start = None
        ind += 1
    if start is not None:
        ranges.append((start, ind))
    return ranges


def b_spline_smooth(_trajectory, vis=False, name="test2.png", return_params=False, removed=None):
    """
    b spline smoothing for missing values (denoted None)
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

    if vis:
        plt.plot(control_points_time, control_points, "ob")
        plt.plot(not_there, [interpolate.splev(du, tck) for du in not_there], "or")

        plt.plot(np.linspace(0, len(_trajectory), 1000),
                 [interpolate.splev(du, tck) for du in np.linspace(0, len(_trajectory), 1000)], "y")

        if removed is not None:
            plt.plot([du[0] for du in removed], [du[1] for du in removed], "oy")
        plt.xlabel("time")
        plt.ylabel("position")
        plt.legend(["s. points", "points", "missing points", "curve", "s. curve"], prop={'size': 10})
        plt.savefig(f'{name}', dpi=300)
        plt.close()
    if return_params:
        return tck
    values = [interpolate.splev(du, tck) for du in np.linspace(0, len(_trajectory), len(_trajectory))]
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
    pcd = open3d.io.read_point_cloud("data_heavy/point_cloud.txt", format='xyzrgb')

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(0.0, 0.0)
        return False

    open3d.visualization.draw_geometries_with_animation_callback([pcd], rotate_view)


def normalize(res, ref):
    inp = res.copy()
    inp[:, 0] = (np.max(ref[:, 0]) - np.min(ref[:, 0]))*(inp[:, 0] - np.min(inp[:, 0]))/(np.max(inp[:, 0]) - np.min(inp[:, 0])) + np.min(ref[:, 0])
    inp[:, 1] = (np.max(ref[:, 1]) - np.min(ref[:, 1]))*(inp[:, 1] - np.min(inp[:, 1]))/(np.max(inp[:, 1]) - np.min(inp[:, 1])) + np.min(ref[:, 1])
    return inp


def draw_text_to_image(img, text, pos):
    # Write some Text
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 4
    fontColor              = (0,0,0)
    lineType               = 2

    cv2.putText(img,text,
        pos,
        font,
        fontScale,
        fontColor,
        lineType)
    return img


def zero_mean(array):
    res = np.zeros_like(array)
    u, v = np.min(array[:, 0]), np.min(array[:, 1])
    res[:, 0] = array[:, 0] - u
    res[:, 1] = array[:, 1] - v
    return res


def draw_image(array):
    array = zero_mean(array)
    res = np.zeros((int(np.max(array[:, 0])+1),
                    int(np.max(array[:, 1])+1), 3))
    for i in range(array.shape[0]):
        u, v = map(int, array[i])
        res[u, v] = (128, 128, 128)
    return res


