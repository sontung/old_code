import sys
import csv
import cv2
import os
import json
import open3d
import numpy as np
import kmeans1d
import pylab as pl
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from sklearn.linear_model import RANSACRegressor, LinearRegression, LogisticRegression, BayesianRidge
from sklearn.svm import SVR
from scipy import interpolate
from matplotlib import pyplot as plt


def neutralize_head_rot(cpd_computations, head_mask_computations):
    new_rot = []
    for idx, x in enumerate(cpd_computations):
        if x is None:
            new_rot.append(None)
        else:
            head_mask_computations[idx] = 90-head_mask_computations[idx]
            new_rot.append(min([x+90, x-90, x], key=lambda du: abs(du-head_mask_computations[idx])))
            print(idx, new_rot[-1],
                  head_mask_computations[idx],
                  [x + 90, x - 90, x],
                  min([x+90, x-90, x], key=lambda du: abs(du-head_mask_computations[idx])),
                  )
    plt.plot(cpd_computations, "r")
    plt.plot(new_rot, "b")
    plt.plot(head_mask_computations, "g")
    plt.legend(["ori", "neutral", "by head mask"])
    plt.savefig("neutral.png")
    plt.close()
    return new_rot


def cpd_smooth():
    angles = [-3.4462777027866776, -2.1150250755250406, -1.8560732528568555, -1.3574491120418282, -1.9546781171926215, -3.2734472716413046, -1.0872772653182894, -3.5105807534448528, -1.630900699090774, -0.7040869824998134, -2.4389707939738656, -1.7013383263965176, -1.700699473614145, -3.2926932466343324, -2.682874266060129, -1.976978904131933, -3.255259406658046, -3.1089576073968206, -3.0907377835695913, -3.9150016407316484, -2.608541079583464, -3.1756303516994775, -3.576447407041296, -3.5430354249761327, -1.7217838073417828, -2.7387528621254464, -2.982519293322194, -1.7270068806145518, -0.14698977255068216, 3.152358527307376, 6.110672974270201, 11.282876184188087, 16.731277354001755, 19.247687601717978, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 22.688711753964206, 6.163171894580451, 5.287295527264798, 14.621030445863529, 24.241563554687293, 26.978299546281132, 27.72163226221281, -39.679278262916284, 26.182223906695505, 12.20184900439492, 17.931186169334115, 13.889772675802327, 8.420033492333705, None, 2.432931184129019, -1.7305925821014478, 0.35896367442838506, -3.6254288311620755, -11.125160058885974, -12.628546710957924, -15.747831931260873, -19.597861043749145, -22.81247066292442, -25.58436698968336, -28.538474775553823, -25.3960300152347, -16.456842931264294, 6.685207752747491, 19.488347538793196, 27.551517685662347, 26.075859374935472, 27.062517178012236, 6.011572279737814, 26.892404943686618, -63.28764972386039, 20.558372192743466, 22.993873603607625, 7.7387799841224005]
    inds = [du for du in range(len(angles)) if angles[du] is not None]
    observations = [du for du in angles if du is not None]

    ear = cv2.imread("../data/curve.png")
    ear = cv2.resize(ear, (ear.shape[1]//4, ear.shape[0]//4))

    nonzero_indices = np.nonzero(ear)
    with open("../data/curve.txt", "w") as fp:
        for i in range(nonzero_indices[0].shape[0]):
            print(nonzero_indices[0][i],
                  nonzero_indices[1][i], file=fp)
    y_data = np.loadtxt('../data/curve.txt')
    x_data = np.array([inds, observations])
    print(y_data.shape)
    plt.plot(y_data[:, 1], y_data[:, 0], "ob")
    plt.plot(x_data[:, 1], x_data[:, 0], "or")

    plt.show()

def normalize_angle(x):
    x = np.radians(x)
    x = x % (2 * np.pi)
    if x > np.pi:
        x -= 2 * np.pi
    return np.degrees(x)

def kalman_smooth():
    angles = [-3.4462777027866776, -2.1150250755250406, -1.8560732528568555, -1.3574491120418282, -1.9546781171926215, -3.2734472716413046, -1.0872772653182894, -3.5105807534448528, -1.630900699090774, -0.7040869824998134, -2.4389707939738656, -1.7013383263965176, -1.700699473614145, -3.2926932466343324, -2.682874266060129, -1.976978904131933, -3.255259406658046, -3.1089576073968206, -3.0907377835695913, -3.9150016407316484, -2.608541079583464, -3.1756303516994775, -3.576447407041296, -3.5430354249761327, -1.7217838073417828, -2.7387528621254464, -2.982519293322194, -1.7270068806145518, -0.14698977255068216, 3.152358527307376, 6.110672974270201, 11.282876184188087, 16.731277354001755, 19.247687601717978, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 22.688711753964206, 6.163171894580451, 5.287295527264798, 14.621030445863529, 24.241563554687293, 26.978299546281132, 27.72163226221281, -39.679278262916284, 26.182223906695505, 12.20184900439492, 17.931186169334115, 13.889772675802327, 8.420033492333705, None, 2.432931184129019, -1.7305925821014478, 0.35896367442838506, -3.6254288311620755, -11.125160058885974, -12.628546710957924, -15.747831931260873, -19.597861043749145, -22.81247066292442, -25.58436698968336, -28.538474775553823, -25.3960300152347, -16.456842931264294, 6.685207752747491, 19.488347538793196, 27.551517685662347, 26.075859374935472, 27.062517178012236, 6.011572279737814, 26.892404943686618, -63.28764972386039, 20.558372192743466, 22.993873603607625, 7.7387799841224005]
    angles_spline = b_spline_smooth(angles)

    inds = [du for du in range(len(angles)) if angles[du] is not None]
    observations = [du for du in angles if du is not None]
    states = [angles[0], 0, 0, 0]
    angles = []
    crash = False
    kp = 0.55
    ki = 0.01
    kd = 0.5
    e_sum = 0
    e_prev = 0 - states[0]
    dt = 1


    # points = SigmaPoints(n=3, alpha=.00001, beta=2, kappa=0,
    #                      subtract=residual_x)
    # ukf = UnscentedKalmanFilter(dim_x=4, dim_z=2, fx=fx, hx=Hx, dt=dt, points=points,
    #           x_mean_fn=state_mean, z_mean_fn=z_mean,
    #           residual_x=residual_x, residual_z=residual_h)

    for i in range(100):
        e = normalize_angle(-states[0])
        if i > 50 and not crash:
            states[3] = 50
            crash = True
        elif i > 50 and crash:
            states[3] = 0
            e_sum += e*dt
            dedt = (e-e_prev)/dt
            u = kp*e+ki*e_sum+kd*dedt
            states[2] = u
            e_prev = e

        if states[3] > 0:
            states[2] = states[3]/2
        states[1] = states[1]+states[2]
        states[0] = states[0]+states[1]
        angles.append(states[0])
        print(-states[0], normalize_angle(-states[0]))

    plt.plot(angles)
    plt.savefig("test.png")


def kalman_smooth2():
    def fx(state_, dt):
        new_state_ = state_[:]
        new_state_[0] = state_[0]+state_[1]*dt
        new_state_[1] = state_[1]+0.5*(state_[2])**2*dt
        new_state_[2] = state_[3]*dt
        new_state_[3] = state_[3]

        return new_state_

    def hx(state_):
        return np.array([state_[0]])

    angles = [-3.4462777027866776, -2.1150250755250406, -1.8560732528568555, -1.3574491120418282, -1.9546781171926215, -3.2734472716413046, -1.0872772653182894, -3.5105807534448528, -1.630900699090774, -0.7040869824998134, -2.4389707939738656, -1.7013383263965176, -1.700699473614145, -3.2926932466343324, -2.682874266060129, -1.976978904131933, -3.255259406658046, -3.1089576073968206, -3.0907377835695913, -3.9150016407316484, -2.608541079583464, -3.1756303516994775, -3.576447407041296, -3.5430354249761327, -1.7217838073417828, -2.7387528621254464, -2.982519293322194, -1.7270068806145518, -0.14698977255068216, 3.152358527307376, 6.110672974270201, 11.282876184188087, 16.731277354001755, 19.247687601717978, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 22.688711753964206, 6.163171894580451, 5.287295527264798, 14.621030445863529, 24.241563554687293, 26.978299546281132, 27.72163226221281, -39.679278262916284, 26.182223906695505, 12.20184900439492, 17.931186169334115, 13.889772675802327, 8.420033492333705, None, 2.432931184129019, -1.7305925821014478, 0.35896367442838506, -3.6254288311620755, -11.125160058885974, -12.628546710957924, -15.747831931260873, -19.597861043749145, -22.81247066292442, -25.58436698968336, -28.538474775553823, -25.3960300152347, -16.456842931264294, 6.685207752747491, 19.488347538793196, 27.551517685662347, 26.075859374935472, 27.062517178012236, 6.011572279737814, 26.892404943686618, -63.28764972386039, 20.558372192743466, 22.993873603607625, 7.7387799841224005]
    angles_spline = b_spline_smooth(angles)

    inds = [du for du in range(len(angles)) if angles[du] is not None]
    observations = [du for du in angles if du is not None]
    points = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=-1)

    f = UnscentedKalmanFilter(dim_x=4, dim_z=1, dt=0.5, hx=hx, fx=fx, points=points)
    f.x = np.array([angles[0], 0., 0., 0.])
    f.P *= 20  # initial uncertainty
    f.R *= 0.5
    # f.Q = Q_discrete_white_noise(dim=4, dt=0.5, var=0.03)

    mu, cov = f.batch_filter(observations)
    (xs, Ps, Ks) = f.rts_smoother(mu, cov)
    for ind, x in enumerate(xs):
        angles[inds[ind]] = x

    # estimate state with filtering and smoothing
    pl.figure()
    ori = pl.plot(angles_spline, color='b')
    rot_smooth_all = pl.plot( b_spline_smooth(angles), color='r')

    pl.legend(
        (ori[0], rot_smooth_all[0]),
        ("ori", 'kalman'),
        loc='best'
    )
    pl.show()


def curve_fit():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures

    angles = [-3.4462777027866776, -2.1150250755250406, -1.8560732528568555, -1.3574491120418282, -1.9546781171926215, -3.2734472716413046, -1.0872772653182894, -3.5105807534448528, -1.630900699090774, -0.7040869824998134, -2.4389707939738656, -1.7013383263965176, -1.700699473614145, -3.2926932466343324, -2.682874266060129, -1.976978904131933, -3.255259406658046, -3.1089576073968206, -3.0907377835695913, -3.9150016407316484, -2.608541079583464, -3.1756303516994775, -3.576447407041296, -3.5430354249761327, -1.7217838073417828, -2.7387528621254464, -2.982519293322194, -1.7270068806145518, -0.14698977255068216, 3.152358527307376, 6.110672974270201, 11.282876184188087, 16.731277354001755, 19.247687601717978, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 22.688711753964206, 6.163171894580451, 5.287295527264798, 14.621030445863529, 24.241563554687293, 26.978299546281132, 27.72163226221281, -39.679278262916284, 26.182223906695505, 12.20184900439492, 17.931186169334115, 13.889772675802327, 8.420033492333705, None, 2.432931184129019, -1.7305925821014478, 0.35896367442838506, -3.6254288311620755, -11.125160058885974, -12.628546710957924, -15.747831931260873, -19.597861043749145, -22.81247066292442, -25.58436698968336, -28.538474775553823, -25.3960300152347, -16.456842931264294, 6.685207752747491, 19.488347538793196, 27.551517685662347, 26.075859374935472, 27.062517178012236, 6.011572279737814, 26.892404943686618, -63.28764972386039, 20.558372192743466, 22.993873603607625, 7.7387799841224005]
    angles_spline = b_spline_smooth(angles)
    observations = np.array([du for du in angles if du is not None]).reshape(-1, 1)
    x = np.array([du for du in range(len(angles)) if angles[du] is not None]).reshape(-1, 1)

    # reg = RANSACRegressor(random_state=0).fit(x, observations)
    # reg = make_pipeline(PolynomialFeatures(3), RANSACRegressor(random_state=0))
    # reg = make_pipeline(PolynomialFeatures(10), RANSACRegressor())
    # reg = SVR(kernel='rbf', C=300, gamma=0.1, epsilon=.1)
    reg = BayesianRidge(tol=1e-6, alpha_init=1., lambda_init=0.001, fit_intercept=False, compute_score=True)

    # reg = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1)
    reg.fit(x, observations)
    test_x = np.linspace(0, x[-1][0], 100).reshape(-1, 1)
    test_y = reg.predict(test_x)

    plt.plot(test_x, test_y, "g")
    plt.plot(x, observations, "ob")
    plt.show()


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
    # grad = np.gradient(np.gradient(path))
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

    ind = 0
    start = None
    end = None
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
    kalman_smooth()
    # kalman_smooth2()
    # dump_into_tracks_osfm()
    # visualize_point_cloud()
    # complement_point_cloud()
