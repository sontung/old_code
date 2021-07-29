import sys
import cv2
import json
import open3d
import numpy as np
import kmeans1d
from scipy import interpolate
from matplotlib import pyplot as plt
from laplacian_fairing_1d import laplacian_fairing


def neutralize_head_rot(cpd_computations, head_mask_computations):
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
    comp1 = [5.028709494712534, -0.6311304744955888, 2.913665770434109, 4.986493018308874, 5.065181388831482, 0.011748543119870029, 4.688225474940464, 4.672185561528583, 4.620775246149054, 4.8387612438699765, 4.800883276821895, 4.983360662767022, 4.471642608297264, 4.73382736145395, 5.057331744886672, 4.789421267194248, 4.989395331222822, 5.032720763497453, 5.047147507678587, 4.912222654160452, 5.033160322779851, 4.844799567079188, 5.028454634833687, 4.888734618983605, 4.790102781833701, 4.294130996907745, 3.784591623096815, 5.035254967746712, 5.588748533822899, 4.122188062500238, 2.863085594784162, 7.599970316295084, 13.611254393927307, 15.73719125356553, 19.574721293490803, 26.930391312890823, 28.601436435834753, None, None, None, None, None, None, None, None, None, None, -4.036308746406928, -23.560784958968505, -35.33577218719292, -42.57621506717781, 41.17597891400469, 35.11123220275485, 28.846005575168064, 23.05700210028158, 16.71299470640557, 11.02755642862374, 5.583176349730191, 0.2666787374502284, -4.168037807028909, -9.035743139627574, -12.46454429396619, -16.97626445059961, -20.880911695723515, -23.193826532921356, -27.48271256104305, -31.89194800740696, -35.22516007390303, -39.43921743526861, -41.79716453920511, 44.305401250833405, 42.11148312801444, 39.566907259025506, 37.60151555685641, 35.78595555880287, 34.07769087419473, 33.28484131479832, 33.51517427036829, 33.10228304175382, 34.48051391684929, 34.97676934069278, 36.93554531959687, 38.84172297430414, 40.5394201876036, 42.47077737903058, 45.702032296140345, -39.96570204618556, -37.224272773739685, -32.42628369572213, -30.115837428639722, -24.364698243216907, -20.08146645631486, -16.338989489180477, -12.679154824683332, -7.052249072494109, -2.4052753791001678, -1.9599939149340102, 5.749111139447085, 10.447306546278325, 13.032653672093693, 18.47863098860658, 1.134874491377857, -16.364168771637864, 30.981232541705786, 5.6021288741745785, -2.114691764539358, 26.263469481504828, 26.606779376363814, 27.669372678320403, 26.319072132530724, 23.640140678297357, 22.742070601868974, 22.43033486434871, 20.426872463172728, 20.207675411602512, 19.272108386069768, 18.118456518619222, 15.114173396005347, 12.98858949619455, 10.998054639977473, 9.97933322821603, 8.151074129870773, 7.026634775063994]
    comp2 = [96.40744028253317, 96.40744028253317, 96.37363936587751, 96.40744028253317, 96.37363936587751, 96.40744028253317, 96.37363936587751, 96.37363936587751, 96.40744028253317, 96.40744028253317, 96.40744028253317, 96.37363936587751, 96.37363936587751, 95.801586887647, 95.801586887647, 95.801586887647, 95.801586887647, 95.801586887647, 95.801586887647, 95.801586887647, 95.801586887647, 96.10468487692889, 95.801586887647, 96.37363936587751, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 97.91551242302658, 90.0, 86.71075732150818, 84.32258333795467, 80.14867265719194, 60.29468760527064, 56.93560254326744, 54.8966719676691, 44.459489812869336, None, None, None, None, 111.03751102542182, 107.5924245621816, 27.62596286734222, 29.699154427550006, 32.9386905429555, 46.67763946615347, 46.8476102659946, 53.46128814257136, 59.09359658141622, 66.31247366327922, 75.32360686254998, 80.26301172959725, 82.95796082627061, 89.00653114371742, 91.60002471514774, 97.16354746583238, 99.00477912527731, 90.0, 91.27303002005671, 96.51101944340064, 99.71056926601348, 105.5086381968298, 109.06257701453984, 111.5953104489677, 112.00585386224785, 112.30620505490764, 137.8271245781613, 141.45555221991668, 135.17521590490534, 146.91514707126504, 146.4430234762542, 140.9293424810033, 140.40833215747537, 141.0795889578385, 142.08328599975255, 143.93770081641026, 144.98846489091767, 62.539775112710934, 60.554571270074405, 148.37675282985143, 145.90502204523446, 143.32565033042684, 140.06801812660518, 134.11313926252836, 133.21008939175394, 126.2538377374448, 125.11859387964981, 116.56505117707799, 99.60520415501296, 95.62240062708973, 90.0, 90.0, 90.0, 94.08561677997488, 92.36137465817562, 90.0, 87.6015786659075, 84.8602426210094, 82.08448757697342, 81.13813559360244, 79.2300117673829, 77.96940390346214, 77.60459316624154, 77.30463526142705, 77.60459316624154, 77.96940390346214, 85.41390110382915, 83.86274405073802, 80.48746239772102, 80.8376529542783, 80.58737056313386, 79.99202019855866, 84.34850511850236, 85.55589387834732, 87.3438527960429, 89.11859600341786, 90.0, 90.0, 90.0, 90.0]

    neutralize_head_rot(comp1, comp2)

