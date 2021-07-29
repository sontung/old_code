import sys
import cv2
import json
import open3d
import numpy as np
import kmeans1d
from scipy import interpolate
from matplotlib import pyplot as plt
from laplacian_fairing_1d import laplacian_fairing


def take_care_near_nones(cpd_computations):
    print("taking care near nones")
    best_solution = cpd_computations[:]
    ranges = partition_by_none(cpd_computations)
    start, end = ranges[0]
    path = cpd_computations[start:end]
    grad = np.diff(path)
    if abs(grad[len(path)-2]/grad[len(path)-3]) > 2:
        print(f" detecting unusual gradient of {round(grad[len(path)-2], 2)}"
              f" which is {round(abs(grad[len(path)-2]/grad[len(path)-3]))} times more"
              f" of prev grad {round(grad[len(path)-3], 2)}")
        path[-1] = None
        best_solution[start:end] = path
    if len(ranges) > 1:
        start, end = ranges[1]
        path = cpd_computations[start:end]
        grad1 = path[0]-path[1]
        grad2 = path[1]-path[2]
        if abs(grad1/grad2) > 2:
            print(f" detecting unusual gradient of {round(grad1, 2)}"
                  f" which is {round(abs(grad1/grad2))} times more"
                  f" of prev grad {round(grad2, 2)}")
            path[0] = None
            best_solution[start:end] = path
    return best_solution


def neutralize_head_rot(cpd_computations, head_mask_computations):
    cpd_computations = take_care_near_nones(cpd_computations)
    for idx, val in enumerate(cpd_computations):
        if val is None:
            head_mask_computations[idx] = None
    head_mask_computations = [90-du if du is not None else None for du in head_mask_computations]
    head_mask_computations = look_for_abnormals(head_mask_computations)

    best_solution = cpd_computations[:]
    print("begin to enforce smoothness")
    ranges = partition_by_none(cpd_computations)
    for start, end in ranges:
        cpd_path = cpd_computations[start:end]
        head_path = head_mask_computations[start:end]
        best_solution[start:end] = smooth_enforce(cpd_path, head_path)

    plt.plot(cpd_computations, "r")
    plt.plot(best_solution, "b")
    plt.plot(head_mask_computations, "g")
    plt.legend(["cpd", "best", "head mask"])
    plt.savefig("neutral.png")
    plt.close()
    return best_solution


def smooth_enforce(path1, path2):
    """
    select the smoothest solution
    Args:
        path1:
        path2:

    Returns:

    """
    assert None not in path1 and None not in path2
    assert len(path2) == len(path1)
    solution1 = [path1[0]]
    solution2 = [path2[0]]

    for x in range(1, len(path1)):
        possible_comp = [path1[x], path2[x], path1[x]-90, path2[x]-90, path1[x]+90, path2[x]+90]
        solution1.append(min(possible_comp, key=lambda du: abs(du-solution1[x-1])))
        solution2.append(min(possible_comp, key=lambda du: abs(du - solution2[x - 1])))
    grad1 = np.sum(np.abs(np.gradient(np.gradient(solution1))))
    grad2 = np.sum(np.abs(np.gradient(np.gradient(solution2))))
    if grad1 < grad2:
        print(f" solution1: using cpd, from {grad1} and {grad2}, we select {grad1}")
        return solution1
    else:
        print(f" solution2: using head mask, from {grad1} and {grad2}, we select {grad2}")
        return solution2


def grad_diff_compute(path, idx):
    grad = np.abs(np.diff(path))
    if grad[idx] <= 0.0001 or np.mean(grad[:idx]) <= 0.0001:
        return grad[idx]
    return grad[idx] / np.mean(grad[:idx])


def look_for_abnormals(rot_computation):
    total_grad1 = 0
    total_grad2 = 0
    ori_comp = rot_computation[:]
    ranges = partition_by_none(rot_computation)
    changed = False
    for start, end in ranges:
        if end-start < 2:
            continue
        path = rot_computation[start: end]
        ori_path = path[:]
        total_grad1 += np.sum(np.abs(np.gradient(np.gradient(path))))
        grad2 = np.abs(np.diff(path))
        avg_grad = []
        for idx in range(len(path[:-1])):
            if idx == 0 or path[idx] is None:
                continue
            grad_diff = grad_diff_compute(path, idx)
            if path[idx] != ori_path[idx]:
                print(" changed", idx, path[idx], ori_path[idx], ori_path[idx-1], grad_diff)

            if grad_diff > 2:
                new_path = path[:]
                new_path[idx+1] = path[idx+1]-90
                new_grad_diff = grad_diff_compute(new_path, idx)
                if new_grad_diff < grad_diff:
                    path[idx + 1] = path[idx + 1] - 90
                    changed = True
            avg_grad.append(grad2[idx])
        total_grad2 += np.sum(np.abs(np.gradient(np.gradient(path))))
        rot_computation[start: end] = path
    if changed:
        print(f"reducing grad from {total_grad1} to {total_grad2}")
        plt.plot(ori_comp)
        plt.plot(rot_computation)
        plt.legend(["ori", "new"])
        plt.savefig("abnormal_head.png")
        plt.close()
    return rot_computation


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


def b_spline_smooth(_trajectory, vis=False, name="test2.png", return_params=False, removed=None):
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
    control_points_smoothed = smooth_1d(control_points, window_len=6)

    tck_smoothed = interpolate.splrep(control_points_time, control_points_smoothed, k=2)
    tck = interpolate.splrep(control_points_time, control_points, k=3)

    if vis:
        plt.plot(control_points_time, control_points_smoothed, "og")
        plt.plot(control_points_time, control_points, "ob")
        plt.plot(not_there, [interpolate.splev(du, tck) for du in not_there], "or")

        plt.plot(np.linspace(0, len(_trajectory), 1000),
                 [interpolate.splev(du, tck) for du in np.linspace(0, len(_trajectory), 1000)], "y")

        plt.plot(np.linspace(0, len(_trajectory), 1000),
                 [interpolate.splev(du, tck_smoothed) for du in np.linspace(0, len(_trajectory), 1000)], "green")

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


def normalize(inp, ref):
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


def remove_condition(path):
    grad = np.gradient(path, 2)
    clusters, centroids = kmeans1d.cluster(grad, 2)
    if clusters.count(0) > clusters.count(1):
        remove_label = 1
    else:
        remove_label = 0
    res = path[:]
    removed_instances = []
    for i in range(len(path)):
        if clusters[i] == remove_label:
            res[i] = None
            removed_instances.append((i, path[i]))
    return res, removed_instances


def refine_path_computation(path, return_removed=False):
    ranges = partition_by_none(path)
    res = path[:]
    removed_instances_all = []
    for start, end in ranges:
        res[start: end], removed_instances = remove_condition(path[start: end])
        for v, k in removed_instances:
            removed_instances_all.append((v+start, k))
    if return_removed:
        print("removing", removed_instances_all)
        return res, removed_instances_all
    return res


def get_translation_scale():
    sys.stdin = open("../data_heavy/frame2ab.txt")
    lines2 = [du[:-1] for du in sys.stdin.readlines()]
    frame2ab = {du.split(" ")[0]: du for du in lines2}
    head_area_img = float(frame2ab["1-1.png"].split(" ")[2])
    head_im = cv2.imread("../data_heavy/area_compute/head-0.png")
    return np.sqrt(head_area_img/np.sum(head_im[:, :, 0]!=255))


def smooth_1d(x, window_len=4, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    y = y[int(window_len/2-1):-int(window_len/2)]
    return list(y)


if __name__ == '__main__':
    comp = [-0.9423589331120863, -1.061323141561521, -0.4817083861581547, -1.0607292262811319, -2.374943394341507, -2.9247544398051897, -2.7226664004464936, -1.7812174668655851, -1.6646302280506404, -2.1718140047749857, -2.545234851650651, -1.728235111323555, -0.6063853369051814, -1.5063452634551158, -1.170761955143116, -0.5433326241408118, -0.26520265104887975, -1.6340720757853344, -1.4102797288891196, -2.0718780222134376, -0.4643702920153006, 1.200890140942879, 0.24455293234167705, -5.03241909577697, -1.2548472579609489, -1.2259716625006312, -0.2893166233784473, -0.060799810425414894, -0.43978791683085133, -1.673523751678178, -1.131805160822326, 1.663723833335142, 0.7978246450488325, 5.5178393456481665, -21.81743457564633, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    take_care_near_nones(comp)
