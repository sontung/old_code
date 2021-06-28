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
from rec_utils import b_spline_smooth, normalize
from tqdm import tqdm
from solve_airbag import compute_ab_pose, compute_ab_frames
from pycpd import RigidRegistration
import matplotlib.pyplot as plt
import time
import argparse

DEBUG_MODE = False


def remove_condition(path):
    grad = np.gradient(path, 2)
    clusters, centroids = kmeans1d.cluster(grad, 2)
    if clusters.count(0) > clusters.count(1):
        remove_label = 1
    else:
        remove_label = 0
    res = path[:]
    for i in range(len(path)):
        if clusters[i] == remove_label:
            res[i] = None
    return res


def refine_path_computation(path):
    res = path[:]
    null_indices = [du for du in range(len(path)) if path[du] is None]
    if len(null_indices) <= 2:
        return res
    res[:min(null_indices)] = remove_condition(path[:min(null_indices)])
    res[max(null_indices)+1:] = remove_condition(path[max(null_indices)+1:])
    return res


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
    for idx in tqdm(range(len(x_traj)), desc="Computing head x-y translation"):
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


def compute_rotation_accurate(reverse_for_vis=False, debugging=DEBUG_MODE):
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
        plt.savefig("../data_heavy/rigid_head_rotation/1-%s.png" % (idx))
        plt.close(fig)

        if debugging:
            print("%s/1-%s.png" % (images_dir, idx), all_angles)
            cv2.imshow("t", img)
            cv2.waitKey()
            cv2.destroyAllWindows()
    stamp = time.time()
    b_spline_smooth(all_angles, vis=True, name=f"rot_ori_{stamp}.png")
    all_angles = refine_path_computation(all_angles)
    all_angles = b_spline_smooth(all_angles, vis=True, name=f"rot_smooth_{stamp}.png")
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
        #rotated_trajectory = None
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


def draw_text_to_image(img, text):
    # Write some Text
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale              = 4 
    fontColor              = (0,0,0)
    lineType               = 2

    cv2.putText(img,text,
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType)
    return img


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


def test():
    angles = [-4.249123672419037, -4.06042588305064, -4.100706659193889, -4.502854905981135, -5.211328649500481, -4.236818542437244, -4.973425832429783, -5.122461669680074, -4.213002968916646, -3.6060948115779814, -3.8728006895552087, -3.801343832294335, -3.298066161415771, -3.6287874176276578, -3.1693856468991584, -3.510242578718621, -3.140791882368664, -4.103912366201475, -4.135883918800803, -4.024808394187761, -4.2437274197720685, -4.463149537001311, -4.208010409803948, -4.851680774004157, -4.1360025643413865, -4.719767764830964, -5.899437021478936, -5.66559692292461, -3.0361637259366314, -0.23594728146812557, 6.26483765369673, 10.299016359770043, 13.943018508997634, -5.813251142483006, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 5.405104832598654, 19.58789686826146, 20.468786114775167, 14.697888651019413, 11.20568145258964, 6.22562999073554, 4.128373419245486, 0.9092651808452706, -0.9603686950307418, -3.197679409451433, -4.320184105038989, -6.299168012111577, -8.77646369575391, -9.53550345993486, -10.972149596051482, -11.560929975665141, -13.128541600299346, -13.063096628992412, -13.670482107365373, -14.218429423869052, -14.15144475912785, -13.911124970738571, -13.06703927110693, -13.02563889257553, -12.571935248126767, -11.687513382410593, -11.207924547575049, -11.290370750130418, -11.556150089107536, -10.401508508456763, -10.146024235612684, -10.565632920824264, -10.711420519845431, -10.563194831318992, -9.986432414128881, -10.160312548723827, -9.055435962078626, -9.477542090709312, -9.982358727296557, -10.08329873016015, -9.749719576922745, -9.315396040761426, -9.1618467150131, -8.662547813203934, -8.099830341997585, -7.310695357768626, -6.851086186580008, -6.572821246436068, -6.78676265174479, -6.495933534014509, -7.847888469525019, -7.871362678973132, -8.466897314198885, -9.400673424387715, -10.419822784460676, -10.442531980976932, -10.07937149928611]
    # all_angles = b_spline_smooth(angles, vis=True, name="bspline")
    x = []
    y = []
    for i, v in enumerate(angles):
        if v is not None:
            x.append(i)
            y.append(v)

    from scipy import interpolate
    x = np.array(x)
    y = np.array(y)
    f = interpolate.Rbf(x, y,
                        function="inverse", mode="1-D")

    new_x = np.array(range(len(angles)))
    new_y = f(new_x)

    plt.plot(x, y, 'o', new_x, new_y)
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', type=bool, default=False, help='Debug mode')
    args = vars(parser.parse_args())

    DEBUG_MODE = args['debug']

    test()
    # visualize(False)
