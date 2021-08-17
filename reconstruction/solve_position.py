import argparse
import glob
import math
import os
import pickle
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from scipy.spatial.transform import Rotation as rot_mat_compute
from tqdm import tqdm
from anomaly_detetion import neutralize_head_rot
from anomaly_detetion import look_for_abnormals_based_on_ear_sizes_tight
from custom_rigid_cpd import RigidRegistration
from laplacian_fairing_1d import laplacian_fairing
from rec_utils import normalize, draw_text_to_image, partition_by_not_none
from solve_airbag import compute_ab_pose, compute_ab_frames, compute_head_ab_areas_image_space
from translation_bound_processing import check_translation_bound
from test_model import new_model, new_model_no_normal


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', type=bool, default=False, help='Debug mode')
parser.add_argument('-f', '--fast', type=bool, default=False, help='Fast mode')

args = vars(parser.parse_args())
DEBUG_MODE = args['debug']
FAST_MODE = args['fast']

# Global variables for directories
FRAMES_INFO_DIR = "../data_heavy/frames/info.txt"

if DEBUG_MODE:
    # shutil.rmtree("test", ignore_errors=True)
    # shutil.rmtree("test2", ignore_errors=True)
    os.makedirs("test", exist_ok=True)
    os.makedirs("test2", exist_ok=True)
    print("running in debug mode")
if FAST_MODE:
    print("running in fast mode (not recommended)")


def visualize_rigid_registration(iteration, error, x, y, ax):
    plt.cla()
    ax.scatter(x[:, 0],  x[:, 1], color='red', label='Target')
    ax.scatter(y[:, 0],  y[:, 1], color='blue', label='Source')
    plt.text(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
        iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)


def compute_translation(ab_transx, ab_transy):
    """
    translation of the head
    inputs:
    ab_transx: float
    ab_transy: float

    outputs:
    trajectories: list,
    x_traj: list,
    y_traj: list,
    ab_transx_new: float,
    ab_transy_new: float
    """
    sys.stdin = open(FRAMES_INFO_DIR)
    lines = [du[:-1] for du in sys.stdin.readlines()]
    all_pixel_dir = "../data_heavy/frames_ear_coord_only"

    dim_x, dim_y = cv2.imread("../data_heavy/frames/1-1.png").shape[:2]

    sys.stdin = open("../data_heavy/frame2ab.txt")
    lines2 = [du[:-1] for du in sys.stdin.readlines()]
    frame2ab = {du.split(" ")[0]: du for du in lines2}

    trajectories = []
    x_traj = []
    y_traj = []
    first_disappear = None
    start_counting = False
    for idx in lines:
        _, _, head_pixels = frame2ab[f"1-{idx}.png"].split(" ")[:3]
        head_pixels = int(head_pixels)

        if head_pixels <= 2 and first_disappear is None:
            first_disappear = [idx, idx]
            start_counting = True
        elif start_counting and head_pixels <= 2:
            first_disappear[1] = idx
        elif start_counting and head_pixels > 2:
            start_counting = False

        with open("%s/1-%s.png" % (all_pixel_dir, idx), "rb") as fp:
            right_pixels_all = pickle.load(fp)
        if len(right_pixels_all) == 0:
            mean = [None, None]
        else:
            mean = np.mean(np.array(right_pixels_all), axis=0)  # origin is top left of window
            mean[0] /= dim_x
            mean[1] /= dim_y
        x_traj.append(mean[0])  # up/down
        y_traj.append(mean[1])  # left/right

    x_traj = look_for_abnormals_based_on_ear_sizes_tight(x_traj)
    y_traj = look_for_abnormals_based_on_ear_sizes_tight(y_traj)

    first_disappear_by_head_masks = list(map(int, first_disappear))
    s0, e0 = first_disappear_by_head_masks
    ranges = [(s, e, abs(s-s0)+abs(e-e0)+abs(e-s-e0+s0)) for s, e in partition_by_not_none(x_traj)]
    first_disappear = min(ranges, key=lambda du: du[-1])
    print(f"from {ranges}, selecting {first_disappear} as closest from {first_disappear_by_head_masks}")
    first_disappear = first_disappear[:2]
    print(" detecting head into airbag between", first_disappear)

    x_traj = laplacian_fairing(x_traj, collision_interval=first_disappear)
    y_traj = laplacian_fairing(y_traj, collision_interval=first_disappear)

    # main
    prev_pos = None
    for idx in range(len(x_traj)):
        mean = np.array([x_traj[idx], y_traj[idx]])
        if prev_pos is not None:
            trans = np.zeros((3, 1))
            move = mean - prev_pos
            trans[2] = -move[1]
            trans[1] = -move[0]
            trajectories.append(trans)
        prev_pos = mean

    ab_transx_new = ab_transx/dim_x-x_traj[0]
    ab_transy_new = ab_transy/dim_y-y_traj[0]

    if first_disappear is not None and int(first_disappear[1]) - int(first_disappear[0]) <= 5:
        first_disappear = None
    trajectories = check_translation_bound(trajectories, ab_transx_new,
                                           ab_transy_new, first_disappear,
                                           dim_x_reproject=540, dim_y_reproject=960)
    # sys.exit()
    print(f"from {ab_transx_new, ab_transy_new} to {ab_transx_new*540, ab_transy_new*960}")
    return trajectories, x_traj, y_traj, ab_transx_new*540, ab_transy_new*960


def rigid_cpd_process(ear_size_filtering, debug=False):
    sys.stdin = open(FRAMES_INFO_DIR)
    lines = [du[:-1] for du in sys.stdin.readlines()]
    images_dir = "../data_heavy/line_images"
    all_angles = []

    source_matrix = np.loadtxt('../data/ear.txt')

    for idx in tqdm(lines, desc="Computing head x-y rotation using rigid CPD"):
        img = cv2.imread(f"{images_dir}/1-{idx}.png")
        if img is None or np.sum(img) == 0:
            all_angles.append(None)
            continue

        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        nonzero_indices = np.nonzero(image)
        target_matrix = np.zeros((nonzero_indices[0].shape[0], 2))
        for i in range(nonzero_indices[0].shape[0]):
            target_matrix[i] = [nonzero_indices[0][i], nonzero_indices[1][i]]
        source_matrix_norm = normalize(source_matrix, target_matrix)

        if not ear_size_filtering:
            all_angles.append(None)
            continue

        reg = RigidRegistration(**{'X': target_matrix, 'Y': source_matrix_norm}, max_iterations=500)
        y_data_norm, (_, rot_mat, _), nb_iter = reg.register()
        rot_angle = np.rad2deg(np.arctan2(rot_mat[1, 0], rot_mat[0, 0]))
        all_angles.append(rot_angle)

        if debug:
            fig = plt.figure()
            fig.add_axes([0, 0, 1, 1])
            ax = fig.axes[0]
            ax.scatter(target_matrix[:, 0], target_matrix[:, 1], color='red', label='Target')
            ax.scatter(y_data_norm[:, 0], y_data_norm[:, 1], color='yellow', label='Source')
            ax.legend(loc='upper left', fontsize='x-large')
            plt.savefig(f"../data_heavy/rigid_head_rotation/1-{idx}.png")
            plt.close(fig)
    return all_angles


def compute_rotation_accurate():
    """
    compute rotation using CPD algorithm
    """
    trajectories = []
    prev_pos = None
    os.makedirs("../data_heavy/rigid_head_rotation", exist_ok=True)
    sys.stdin = open(FRAMES_INFO_DIR)
    lines = [du[:-1] for du in sys.stdin.readlines()]
    ear_size_filtering = look_for_abnormals_based_on_ear_sizes_tight([0 for _ in lines], return_selections=True)

    if not FAST_MODE:
        all_angles = rigid_cpd_process(ear_size_filtering)
        ori_angles = all_angles[:]
        all_angles = neutralize_head_rot(ori_angles, compute_rotation()[-1])
        cpd_angles = laplacian_fairing(ori_angles)
        all_angles_before_null = all_angles[:]
        all_angles = laplacian_fairing(all_angles)
    else:
        all_angles = compute_rotation()[-1]
        all_angles_before_null = all_angles[:]
        all_angles = laplacian_fairing(all_angles)
        cpd_angles = all_angles[:]

    for rot_deg_overall in all_angles:
        if prev_pos is not None:
            move = rot_deg_overall - prev_pos
            trajectories.append(-move)
        prev_pos = rot_deg_overall
    outputs = {
        "trajectories": trajectories,
        "all_angles": all_angles,
        "all_angles_before_null": all_angles_before_null,
        "cpd_angles": cpd_angles,
        "ear_size_filtering": ear_size_filtering,
    }
    return outputs


def compute_rotation(view=1):
    """
    compute rotation by OOB of head masks
    """
    sys.stdin = open(FRAMES_INFO_DIR)
    trajectories = []
    prev_pos = None

    lines = [du[:-1] for du in sys.stdin.readlines()]
    sys.stdin = open("../data_heavy/frame2ab.txt")
    lines2 = [du[:-1] for du in sys.stdin.readlines()]
    frame2ab = {du.split(" ")[0]: du for du in lines2}

    # check view exist
    all_key = list(frame2ab.keys())
    views = [int(key.split('-')[0]) for key in all_key]
    if view not in views:
        print(f"View {view} does not exist")
        return None, [], []

    rot_all = []
    for frn in lines:
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

    all_angles_before_null = rot_all[:]
    all_angles = laplacian_fairing(rot_all)

    for rot_deg_overall in all_angles:
        if prev_pos is not None:
            move = rot_deg_overall - prev_pos
            trajectories.append(-move)
        prev_pos = rot_deg_overall
    return trajectories, all_angles, all_angles_before_null


def sim_prior_view(trajectory, rotated_trajectory, ab_transx2, ab_transy2, head_scaling_additional=None):
    ab_mesh_dir = "../sph_data/mc_solutions_smoothed"
    os.makedirs("../data_heavy/area_compute/", exist_ok=True)
    pcd = new_model_no_normal()
    pcd.translate([0, 0, 0], relative=False)
    ab_scale, ab_transy, ab_transx, ab_rot, ab_area, head_area = compute_ab_pose()

    # this is just some inaccurate scaling for getting the sizes of both head and ab equal
    global_scale_ab_list = []
    for ab_dir in glob.glob(f"{ab_mesh_dir}/*"):
        ab = o3d.io.read_triangle_mesh(ab_dir)
        scale1 = pcd.get_surface_area() / ab.get_surface_area()
        global_scale_ab_list.append(math.sqrt(scale1 / ab_scale))
    global_scale_ab = np.mean(global_scale_ab_list)

    start_ab, _ = compute_ab_frames()
    ab_counter = 0
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.get_view_control().set_zoom(1.5)
    arr = []

    if head_scaling_additional is not None:
        pcd.scale(head_scaling_additional, pcd.get_center())
    else:
        print(f"First sim - airbag pose: translation=({ab_transx2}, {ab_transy2}), rotation={ab_rot}")

    for counter in range(len(trajectory)):
        ab_added = False
        pcd.translate(trajectory[counter % len(trajectory)])
        rot_mat = rot_mat_compute.from_euler('x', rotated_trajectory[counter],
                                             degrees=True).as_matrix()
        pcd.rotate(rot_mat, pcd.get_center())

        vis.update_geometry(pcd)

        vis.get_view_control().rotate(-500, 0)
        vis.capture_screen_image("../data_heavy/area_compute/head-%s.png" % counter, do_render=True)
        ab = None
        if counter >= start_ab-1:
            ab = o3d.io.read_triangle_mesh(f"{ab_mesh_dir}/new_particles_{ab_counter}.obj")
            arr.append(f"{ab_mesh_dir}/new_particles_{ab_counter}.obj")
            ab.scale(global_scale_ab, ab.get_center())
            ab.translate([0, -ab_transx2, -ab_transy2])
            ab.rotate(rot_mat_compute.from_euler("y", 90, degrees=True).as_matrix())
            ab.rotate(rot_mat_compute.from_euler("x", -90+ab_rot, degrees=True).as_matrix())
            ab_added = True
            vis.remove_geometry(pcd, reset_bounding_box=False)
            vis.add_geometry(ab, reset_bounding_box=False)
            vis.capture_screen_image("../data_heavy/area_compute/ab-%s.png" % counter, do_render=True)
            vis.add_geometry(pcd, reset_bounding_box=False)
            ab_counter += 1

        if ab_added and ab is not None:
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
    head_area = np.max(arr)

    image_names = glob.glob("../data_heavy/area_compute/ab*")
    arr = []
    for name in image_names:
        im = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        arr.append(np.sum(im != 255))
    ab_area = np.max(arr)
    return head_area, ab_area


def compute_head_ab_areas():
    """
    prior simulation to compute head/ab scale ratio
    """
    ab_scale, ab_transy, ab_transx, ab_rot, ab_area, head_area = compute_ab_pose()
    trajectory, ne_trans_x_traj, ne_trans_y_traj, ab_transx2, ab_transy2 = compute_translation(ab_transx, ab_transy)
    computations = compute_rotation_accurate()

    rotated_trajectory = computations["trajectories"]
    ne_rot_traj = computations["all_angles"]
    all_angles_before_null = computations["all_angles_before_null"]
    cpd_angles = computations["cpd_angles"]
    ear_size_filtering = computations["ear_size_filtering"]

    assert len(trajectory) == len(rotated_trajectory)

    head_area, ab_area = sim_prior_view(trajectory, rotated_trajectory, ab_transx2, ab_transy2)

    scale1 = head_area/ab_area
    print(f" first pass: head/ab = {scale1} (ref. = {ab_scale})")
    if ab_scale >= 0.27:
        head_scale = np.sqrt(0.27/scale1)
    else:
        head_scale = np.sqrt(ab_scale/scale1)

    head_area, ab_area = sim_prior_view(trajectory, rotated_trajectory, ab_transx2, ab_transy2,
                                        head_scaling_additional=head_scale)
    print(f" second pass: head/ab = {head_area / ab_area} (ref. = {ab_scale})")

    results = {"head scale": head_scale,
               "trajectory": trajectory,
               "rot trajectory": rotated_trajectory,
               "ne rot traj": ne_rot_traj,
               "all angles before null": all_angles_before_null,
               "cpd angles": cpd_angles,
               "ear size filter": ear_size_filtering}
    results2 = [ab_scale, ab_transx2, ab_transy2, ab_rot]
    os.makedirs("../data_const/final_vis", exist_ok=True)
    return results, results2, (ab_transx, ab_transy)


def visualize(debug_mode=DEBUG_MODE):
    ab_mesh_dir = "../sph_data/mc_solutions_smoothed"
    os.makedirs("../data_heavy/saved/", exist_ok=True)
    pcd = new_model()
    pcd.translate([0, 0, 0], relative=False)
    pcd.compute_vertex_normals()
    du_outputs, du_outputs2, (ab_transx_ori, ab_transy_ori) = compute_head_ab_areas()
    ab_scale, ab_transx, ab_transy, ab_rot = du_outputs2

    trajectory = du_outputs["trajectory"]
    rotated_trajectory = du_outputs["rot trajectory"]
    global_head_scale = du_outputs["head scale"]

    # this is just some inaccurate scaling for getting the sizes of both head and ab equal
    global_scale_ab_list = []
    for ab_dir in glob.glob(f"{ab_mesh_dir}/*"):
        ab = o3d.io.read_triangle_mesh(ab_dir)
        scale1 = pcd.get_surface_area() / ab.get_surface_area()
        global_scale_ab_list.append(math.sqrt(scale1 / ab_scale))
    global_ab_scale = np.mean(global_scale_ab_list)

    print(f"Second sim - airbag pose: translation=({ab_transx}, {ab_transy}), rotation="
          f"{ab_rot}, ab scale={global_ab_scale},"
          f" head scale = {global_head_scale}")
    start_ab, _ = compute_ab_frames()
    ab_counter = 0
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.get_view_control().set_zoom(1.5)

    # write computations to disk
    results_to_disk = [global_head_scale, global_ab_scale,
                       trajectory, rotated_trajectory,
                       ab_transx, ab_transy, ab_rot,
                       start_ab]
    os.makedirs("../data_const/final_vis", exist_ok=True)
    with open("../data_const/final_vis/everything_you_need.pkl", "wb") as pickle_file:
        pickle.dump(results_to_disk, pickle_file)

    # start writing visualizations
    trans_actual_traj = []
    rot_actual_traj = []
    pcd.scale(global_head_scale, pcd.get_center())
    head_centers = []
    ab_centers = []

    for counter in tqdm(range(len(trajectory)), desc="Completing simulation"):
        ab_added = False
        pcd.translate(trajectory[counter % len(trajectory)])
        rot_mat = rot_mat_compute.from_euler('x', rotated_trajectory[counter % len(trajectory)],
                                             degrees=True).as_matrix()
        pcd.rotate(rot_mat, pcd.get_center())
        trans_actual_traj.append(trajectory[counter % len(trajectory)])
        rot_actual_traj.append(rotated_trajectory[counter % len(trajectory)])

        # if rotated_trajectory_z is not None:
        #     rot_mat_z = rot_mat_compute.from_euler('z', -rotated_trajectory_z[counter % len(trajectory)],
        #                                            degrees=True).as_matrix()
        #     pcd.rotate(rot_mat_z, pcd.get_center())

        vis.update_geometry(pcd)

        ab = None
        if counter >= start_ab-1:
            ab = o3d.io.read_triangle_mesh(f"{ab_mesh_dir}/new_particles_%d.obj" % ab_counter)
            ab.translate([0, 0, 0], relative=False)
            ab.compute_vertex_normals()
            ab.compute_vertex_normals()
            ab.scale(global_ab_scale, ab.get_center())
            ab.translate([0, -ab_transx, -ab_transy])
            ab.rotate(rot_mat_compute.from_euler("y", 90, degrees=True).as_matrix())
            ab.rotate(rot_mat_compute.from_euler("x", -90+ab_rot, degrees=True).as_matrix())
            ab_added = True
            vis.add_geometry(ab, reset_bounding_box=False)
            ab_counter += 1
            ab_centers.append(ab.get_center())
        else:
            ab_centers.append(None)
        head_centers.append(pcd.get_center())

        vis.get_view_control().rotate(-500, 0)
        vis.capture_screen_image("../data_heavy/saved/v1-%s.png" % counter, do_render=True)

        vis.get_view_control().rotate(500, 0)
        vis.capture_screen_image("../data_heavy/saved/v2-%s.png" % counter, do_render=True)
        if ab_added and ab is not None:
            vis.remove_geometry(ab, reset_bounding_box=False)
    vis.destroy_window()

    if debug_mode:
        ne_rot_traj = du_outputs["ne rot traj"]
        all_angles_before_null = du_outputs["all angles before null"]
        cpd_angles = du_outputs["cpd angles"]
        ear_size_filtering = du_outputs["ear size filter"]

        sys.stdin = open("../data_heavy/head_masks.txt")
        lines2 = [du[:-1] for du in sys.stdin.readlines()]
        frame2head_rect = {du.split(" ")[0]: du for du in lines2}
        head_rotations_by_masks = compute_rotation()[-1]
        os.makedirs("test", exist_ok=True)
        os.makedirs("test2", exist_ok=True)
        sys.stdin = open(FRAMES_INFO_DIR)
        lines = [du[:-1] for du in sys.stdin.readlines()]
        ear_dir = "../data_heavy/frames_ear_only"
        rigid_dir = "../data_heavy/rigid_head_rotation"
        images_dir = "../data_heavy/line_images"

        lines = lines[1:]
        all_angles_before_null = all_angles_before_null[1:]
        ne_rot_traj = ne_rot_traj[1:]
        head_rotations_by_masks = head_rotations_by_masks[1:]
        cpd_angles = cpd_angles[1:]
        ear_size_filtering = ear_size_filtering[1:]

        assert len(lines) == len(trajectory)
        for counter, ind in enumerate(lines):
            line_img = cv2.imread("%s/1-%s.png" % (images_dir, ind))
            ear_img = cv2.imread("%s/1-%s.png" % (ear_dir, ind))
            rigid_img = cv2.imread("%s/1-%s.png" % (rigid_dir, ind))
            if rigid_img is None:
                rigid_img = np.zeros_like(ear_img)

            arr = np.nonzero(ear_img)

            if len(arr[0]) > 0:
                ear_img = ear_img[np.min(arr[0]):np.max(arr[0]), np.min(arr[1]): np.max(arr[1])]
                if ear_img.shape[0] > 0 and ear_img.shape[1] > 0:
                    try:
                        img_size = max([ear_img, line_img, rigid_img], key=lambda x: x.shape[0]*x.shape[1])
                        ear_img = cv2.resize(ear_img, (img_size.shape[1], img_size.shape[0]))
                        line_img = cv2.resize(line_img, (img_size.shape[1], img_size.shape[0]))
                        rigid_img = cv2.resize(rigid_img, (img_size.shape[1], img_size.shape[0]))
    
                        cv2.imwrite(f"test/ear-{ind}.png", np.hstack([ear_img, line_img, rigid_img]))
                    except AttributeError:
                        pass

            im1 = cv2.imread("../data_heavy/saved/v1-%s.png" % counter)
            seg_im1 = cv2.imread('../data_heavy/frames_seg_abh_vis/1-%s.png' % ind)
            ear_im1 = np.zeros_like(seg_im1).astype(np.uint8)
            with open(f"../data_heavy/frames_ear_coord_only/1-{ind}.png", "rb") as ear_file:
                ear = pickle.load(ear_file)
            ear_im1[ear[:, 0], ear[:, 1]] = [125, 60, 80]
            seg_im1 = cv2.addWeighted(seg_im1, 0.3, ear_im1, 0.7, 0)
            x1, y1, x2, y2 = map(int, frame2head_rect[f"1-{ind}.png"].split(" ")[1:])
            cv2.circle(seg_im1, (int(ab_transy_ori), int(ab_transx_ori)), 5, (255, 128, 255), -1)
            cv2.line(seg_im1, (x1, y1), (x2, y2), (255, 255, 255), 5)
            cv2.imwrite(f"../data_heavy/frames_seg_abh/1-{ind}.png", seg_im1)

            seg_im1 = cv2.resize(seg_im1, (im1.shape[1], im1.shape[0]))

            try:
                im_view2 = cv2.imread("../data_heavy/frames/2-%s.png" % ind).astype(np.uint8)
                seg_view2 = cv2.imread('../data_heavy/frames_seg_abh/2-%s.png' % ind).astype(np.uint8)
                seg_im2 = cv2.addWeighted(im_view2, 0.3, seg_view2, 0.7, 0)
                x12, y12, x22, y22 = map(int, frame2head_rect[f"2-{ind}.png"].split(" ")[1:])
                cv2.line(seg_im2, (x12, y12), (x22, y22), (255, 255, 255), 5)
                cv2.imwrite(f"../data_heavy/frames_seg_abh/2-{ind}.png", seg_im2)
                seg_im2 = cv2.resize(seg_im2, (im1.shape[1], im1.shape[0]))

            except AttributeError:
                print(f"../data_heavy/frames/2-{counter}.png does not exist")
                seg_im2 = np.zeros_like(seg_im1)

            info_img = np.ones_like(im1)*255
            info_img = draw_text_to_image(info_img, "rot=%.3f" % (ne_rot_traj[counter]), (100, 100))
            info_img = draw_text_to_image(info_img, f"head pos=(%.2f %.2f)" % (head_centers[counter][1],
                                                                               head_centers[counter][2]), (100, 500))
            if ab_centers[counter] is not None:
                info_img = draw_text_to_image(info_img, f"ab pos2=(%.2f %.2f)" % (ab_centers[counter][1],
                                                                                  ab_centers[counter][2]), (100, 600))
            info_img = draw_text_to_image(info_img, f"ab pos1=(%.2f %.2f)" % (int(ab_transx_ori),
                                                                              int(ab_transy_ori)), (100, 700))

            if head_rotations_by_masks[counter] is not None:
                info_img = draw_text_to_image(info_img, "rot. (head mask) =%.2f" % (-head_rotations_by_masks[counter]+90), (100, 900))
            if cpd_angles[counter] is not None:
                info_img = draw_text_to_image(info_img, "rot. (cpd) =%.2f" % (cpd_angles[counter]), (100, 1000))

            if all_angles_before_null[counter] is None:
                info_img = draw_text_to_image(info_img, "missing comp.", (100, 800))
            info_img = draw_text_to_image(info_img, "trusted="+str(ear_size_filtering[counter]), (100, 200))

            res_im = np.vstack([np.hstack([seg_im1, im1]), np.hstack([seg_im2, info_img])])
            res_im = cv2.resize(res_im, (res_im.shape[1]//2, res_im.shape[0]//2))
            cv2.imwrite(f"test2/res-{ind}.png", res_im)


if __name__ == '__main__':
    visualize()
