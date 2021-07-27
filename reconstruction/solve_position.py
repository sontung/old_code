import os
import math
import glob
import pickle
import sys
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import shutil
import argparse
from scipy.spatial.transform import Rotation as rot_mat_compute
from rec_utils import b_spline_smooth, normalize, draw_text_to_image, neutralize_head_rot
from tqdm import tqdm
from laplacian_fairing_1d import laplacian_fairing
from solve_airbag import compute_ab_pose, compute_ab_frames, compute_head_ab_areas_image_space
from custom_rigid_cpd import RigidRegistration
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', type=bool, default=False, help='Debug mode')
parser.add_argument('-f', '--fast', type=bool, default=False, help='Fast mode')

args = vars(parser.parse_args())
DEBUG_MODE = args['debug']
FAST_MODE = args['fast']

if DEBUG_MODE:
    shutil.rmtree("test", ignore_errors=True)
    shutil.rmtree("test2", ignore_errors=True)
    os.makedirs("test", exist_ok=True)
    os.makedirs("test2", exist_ok=True)
    print("running in debug mode")
if FAST_MODE:
    print("running in fast mode (not recommended)")


def visualize_rigid_registration(iteration, error, X, Y, ax):

    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], color='blue', label='Source')
    plt.text(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
        iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)


def compute_translation(debugging=DEBUG_MODE):
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
            print(f"did not detect ear in 1-{idx}.png")
            mean = [None, None]
        else:
            mean = np.mean(np.array(right_pixels_all), axis=0)
        x_traj.append(mean[0])
        y_traj.append(mean[1])

    # b spline interpolation
    if debugging:
        b_spline_smooth(x_traj, vis=True, name=f"trans_x_ori.png")
        b_spline_smooth(y_traj, vis=True, name=f"trans_y_ori.png")
        x_traj = b_spline_smooth(x_traj, vis=True, name=f"trans_x_refined.png")
        y_traj = b_spline_smooth(y_traj, vis=True, name=f"trans_y_refined.png")
    else:
        x_traj = b_spline_smooth(x_traj)
        y_traj = b_spline_smooth(y_traj)

    for idx in tqdm(range(len(x_traj)), desc="Computing head x-y translation"):
        mean = np.array([x_traj[idx], y_traj[idx]])
        if prev_pos is not None:
            trans = np.zeros((3, 1))
            move = mean - prev_pos
            trans[2] = -move[1]
            trans[1] = -move[0]
            trajectories.append(trans)
        prev_pos = mean
    return trajectories, x_traj, y_traj


def compute_rotation_accurate():
    sys.stdin = open("../data_heavy/frames/info.txt")
    lines = [du[:-1] for du in sys.stdin.readlines()]
    images_dir = "../data_heavy/line_images"
    trajectories = []
    prev_pos = None
    all_angles = []
    os.makedirs("../data_heavy/rigid_head_rotation", exist_ok=True)
    cost_arr = []
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

        # if idx != "78":
        #     continue
        #
        # if idx == "78":
        #     reg = RigidRegistration(**{'X': target_matrix, 'Y': source_matrix_norm}, max_iterations=500)
        #
        #     fig = plt.figure()
        #     fig.add_axes([0, 0, 1, 1])
        #     callback = partial(visualize_rigid_registration, ax=fig.axes[0])
        #     y_data_norm, (_, rot_mat, _) = reg.register(callback)
        #
        #     plt.show()
        #     plt.close(fig)

        reg = RigidRegistration(**{'X': target_matrix, 'Y': source_matrix_norm}, max_iterations=500)
        y_data_norm, (_, rot_mat, _) = reg.register()
        rot_angle = np.rad2deg(np.arctan2(rot_mat[1, 0], rot_mat[0, 0]))
        all_angles.append(rot_angle)

        cost_arr.append(reg.q)

        fig = plt.figure()
        fig.add_axes([0, 0, 1, 1])
        ax = fig.axes[0]
        ax.scatter(target_matrix[:, 0],  target_matrix[:, 1], color='red', label='Target')
        ax.scatter(y_data_norm[:, 0],  y_data_norm[:, 1], color='yellow', label='Source')
        ax.legend(loc='upper left', fontsize='x-large')
        plt.savefig(f"../data_heavy/rigid_head_rotation/1-{idx}.png")
        plt.close(fig)

    ori_angles = all_angles[:]
    all_angles = neutralize_head_rot(ori_angles, compute_rotation()[-1])
    all_angles_before_null = all_angles[:]
    all_angles = laplacian_fairing(all_angles)

    for rot_deg_overall in all_angles:
        if prev_pos is not None:
            move = rot_deg_overall - prev_pos
            trajectories.append(-move)
        prev_pos = rot_deg_overall

    return trajectories, all_angles, all_angles_before_null


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
        return None, [], []

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

    all_angles_before_null = rot_all[:]
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
    return trajectories, all_angles, all_angles_before_null


def apply_limited(ab_position, head_traj, axis):
    total_move = 0
    for i in range(len(head_traj)):
        total_move += head_traj[i][axis]
        if total_move < ab_position:
            total_move = total_move - head_traj[i][axis]
            head_traj[i][axis] = 0

    return head_traj


def compute_head_ab_areas():
    ab_mesh_dir = "../sph_data/mc_solutions_smoothed"
    os.makedirs("../data_heavy/area_compute/", exist_ok=True)
    pcd = o3d.io.read_triangle_mesh("../data/max-planck.obj")
    ab_scale, ab_transx, ab_transy, ab_rot, ab_area, head_area = compute_ab_pose()

    # scale the AB to match the scale head/ab in image space
    global_scale_ab_list = []
    for ab_dir in glob.glob(f"{ab_mesh_dir}/*"):
        ab = o3d.io.read_triangle_mesh(ab_dir)
        scale1 = pcd.get_surface_area() / ab.get_surface_area()
        global_scale_ab_list.append(math.sqrt(scale1 / ab_scale))
    global_scale_ab = np.mean(global_scale_ab_list)

    trajectory, ne_trans_x_traj, ne_trans_y_traj = compute_translation()
    rotated_trajectory_z, _, _ = compute_rotation(view=2)
    if not FAST_MODE:
        rotated_trajectory, ne_rot_traj, all_angles_before_null = compute_rotation_accurate()
    else:
        rotated_trajectory, ne_rot_traj, all_angles_before_null = compute_rotation()

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
    arr = []

    trajectory = apply_limited(-ab_transx, trajectory, axis=1)
    trajectory = apply_limited(-ab_transx, trajectory, axis=2)

    for counter in tqdm(range(len(trajectory)), desc="Prior sim to compute view areas"):
        ab_added = False
        pcd.translate(trajectory[counter % len(trajectory)])
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
        if counter >= start_ab-1:
            ab = o3d.io.read_triangle_mesh(f"{ab_mesh_dir}/new_particles_{ab_counter}.obj")
            arr.append(f"{ab_mesh_dir}/new_particles_{ab_counter}.obj")
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
    head_area = np.max(arr)

    image_names = glob.glob("../data_heavy/area_compute/ab*")
    arr = []
    for name in image_names:
        im = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        arr.append(np.sum(im != 255))
    ab_area = np.max(arr)
    results = [head_area, ab_area, trajectory, rotated_trajectory, rotated_trajectory_z,
               ne_rot_traj, ne_trans_x_traj, ne_trans_y_traj, all_angles_before_null]
    os.makedirs("../data_const/final_vis", exist_ok=True)
    with open("../data_const/final_vis/trajectory.pkl", "wb") as pickle_file:
        pickle.dump(results, pickle_file)
    return results


def visualize(debug_mode=DEBUG_MODE):
    ab_mesh_dir = "../sph_data/mc_solutions_smoothed"
    os.makedirs("../data_heavy/saved/", exist_ok=True)
    pcd = o3d.io.read_triangle_mesh("../data/max-planck.obj")
    pcd.compute_vertex_normals()
    ab_scale, ab_transx, ab_transy, ab_rot, ab_area, head_area = compute_ab_pose()
    du_outputs = compute_head_ab_areas()
    sim_head_area, sim_ab_area, trajectory, rotated_trajectory, \
    rotated_trajectory_z, ne_rot_traj, ne_trans_x_traj, ne_trans_y_traj, all_angles_before_null = du_outputs
    img_ab_area, img_head_area = compute_head_ab_areas_image_space()

    # scale both head and ab to match image space
    global_scale_ab_list = []
    for ab_dir in glob.glob(f"{ab_mesh_dir}/*"):
        ab = o3d.io.read_triangle_mesh(ab_dir)
        scale1 = pcd.get_surface_area() / ab.get_surface_area()
        global_scale_ab_list.append(math.sqrt(scale1 / ab_scale))
    global_ab_scale = np.mean(global_scale_ab_list)

    global_head_scale = np.sqrt(img_head_area/sim_head_area)
    global_ab_scale *= np.sqrt(img_ab_area/sim_ab_area)

    print(f"Airbag pose: translation=({ab_transx}, {ab_transy}), rotation="
          f"{ab_rot}, ab scale={global_ab_scale},"
          f" head scale = {global_head_scale}")
    start_ab, _ = compute_ab_frames()
    ab_counter = 0
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.get_view_control().set_zoom(1.5)

    trans_actual_traj = []
    rot_actual_traj = []
    pcd.scale(global_head_scale, pcd.get_center())
    head_centers = []
    ab_centers = []

    trajectory = apply_limited(-ab_transx, trajectory, axis=1)
    trajectory = apply_limited(-ab_transx, trajectory, axis=2)

    for counter in tqdm(range(len(trajectory)), desc="Completing simulation"):
        ab_added = False
        pcd.translate(trajectory[counter % len(trajectory)])
        rot_mat = rot_mat_compute.from_euler('x', rotated_trajectory[counter % len(trajectory)],
                                             degrees=True).as_matrix()
        pcd.rotate(rot_mat, pcd.get_center())
        trans_actual_traj.append(trajectory[counter % len(trajectory)])
        rot_actual_traj.append(rotated_trajectory[counter % len(trajectory)])

        if rotated_trajectory_z is not None:
            rot_mat_z = rot_mat_compute.from_euler('z', rotated_trajectory_z[counter % len(trajectory)],
                                                   degrees=True).as_matrix()
            pcd.rotate(rot_mat_z, pcd.get_center())

        vis.update_geometry(pcd)

        if counter >= start_ab-1:
            ab = o3d.io.read_triangle_mesh(f"{ab_mesh_dir}/new_particles_%d.obj" % ab_counter)
            ab.compute_vertex_normals()
            ab.scale(global_ab_scale, ab.get_center())
            ab.translate([0, -ab_transy, -ab_transx])
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
        vis.get_view_control().rotate(-200, 0)
        vis.capture_screen_image("../data_heavy/saved/v2-%s.png" % counter, do_render=True)
        vis.get_view_control().rotate(200, 0)
        if ab_added:
            vis.remove_geometry(ab, reset_bounding_box=False)
    vis.destroy_window()

    if debug_mode:
        from solve_airbag import compute_ab_trans

        sys.stdin = open("../data_heavy/head_masks.txt")
        lines2 = [du[:-1] for du in sys.stdin.readlines()]
        frame2head_rect = {du.split(" ")[0]: du for du in lines2}
        head_rotations_by_masks = compute_rotation()[-1]
        ab_trans_image_space = compute_ab_trans()
        os.makedirs("test", exist_ok=True)
        os.makedirs("test2", exist_ok=True)
        sys.stdin = open("../data_heavy/frames/info.txt")
        lines = [du[:-1] for du in sys.stdin.readlines()]
        ear_dir = "../data_heavy/frames_ear_only"
        rigid_dir = "../data_heavy/rigid_head_rotation"
        images_dir = "../data_heavy/line_images"
        ab_trans_image_space_x = ab_trans_image_space[0][1:]
        ab_trans_image_space_y = ab_trans_image_space[1][1:]
        lines = lines[1:]
        all_angles_before_null = all_angles_before_null[1:]
        ne_rot_traj = ne_rot_traj[1:]
        # vis_rotated_trajectory_z = rotated_trajectory_z[1:]
        head_rotations_by_masks = head_rotations_by_masks[1:]
        assert len(lines) == len(trajectory)
        for counter, ind in enumerate(lines):
            line_img = cv2.imread("%s/1-%s.png" % (images_dir, ind))
            ear_img = cv2.imread("%s/1-%s.png" % (ear_dir, ind))
            rigid_img = cv2.imread("%s/1-%s.png" % (rigid_dir, ind))
            arr = np.nonzero(ear_img)
            if len(arr[0]) > 0:
                ear_img = ear_img[np.min(arr[0]):np.max(arr[0]), np.min(arr[1]): np.max(arr[1])]
                img_size = max([ear_img, line_img, rigid_img], key=lambda x: x.shape[0]*x.shape[1])

                ear_img = cv2.resize(ear_img, (img_size.shape[1], img_size.shape[0]))
                line_img = cv2.resize(line_img, (img_size.shape[1], img_size.shape[0]))
                rigid_img = cv2.resize(rigid_img, (img_size.shape[1], img_size.shape[0]))

                cv2.imwrite(f"test/ear-{ind}.png", np.hstack([ear_img, line_img, rigid_img]))
            im1 = cv2.imread("../data_heavy/saved/v1-%s.png" % counter)
            seg_im1 = cv2.imread('../data_heavy/frames_seg_abh_vis/1-%s.png' % ind)
            ear_im1 = np.zeros_like(seg_im1).astype(np.uint8)
            with open(f"../data_heavy/frames_ear_coord_only/1-{ind}.png", "rb") as ear_file:
                ear = pickle.load(ear_file)
            ear_im1[ear[:, 0], ear[:, 1]] = [125, 60, 80]
            seg_im1 = cv2.addWeighted(seg_im1, 0.3, ear_im1, 0.7, 0)
            x1, y1, x2, y2 = map(int, frame2head_rect[f"1-{ind}.png"].split(" ")[1:])
            cv2.circle(seg_im1, (x1, y1), 3, (255, 255, 255), -1)
            cv2.circle(seg_im1, (x2, y2), 3, (255, 255, 255), -1)
            cv2.line(seg_im1, (x1, y1), (x2, y2), (255, 255, 255), 5)
            seg_im1 = cv2.resize(seg_im1, (im1.shape[1], im1.shape[0]))

            try:
                im_view2 = cv2.imread("../data_heavy/frames/2-%s.png" % ind).astype(np.uint8)
                seg_view2 = cv2.imread('../data_heavy/frames_seg_abh/2-%s.png' % ind).astype(np.uint8)
                seg_im2 = cv2.addWeighted(im_view2, 0.3, seg_view2, 0.7, 0)
                x12, y12, x22, y22 = map(int, frame2head_rect[f"2-{ind}.png"].split(" ")[1:])
                cv2.circle(seg_im1, (x12, y12), 3, (255, 255, 255), -1)
                cv2.circle(seg_im2, (x22, y22), 3, (255, 255, 255), -1)
                cv2.line(seg_im2, (x12, y12), (x22, y22), (255, 255, 255), 5)
                seg_im2 = cv2.resize(seg_im2, (im1.shape[1], im1.shape[0]))

            except AttributeError:
                print(f"../data_heavy/frames/2-{counter}.png does not exist")
                seg_im2 = np.zeros_like(seg_im1)

            info_img = np.ones_like(im1)*255
            info_img = draw_text_to_image(info_img, "rot=%.3f" % (ne_rot_traj[counter]), (100, 100))
            info_img = draw_text_to_image(info_img, "trans=%.3f %.3f" % (ne_trans_x_traj[counter],
                                                                         ne_trans_y_traj[counter]), (100, 200))
            info_img = draw_text_to_image(info_img, "act. rot=%.3f" % (rot_actual_traj[counter]), (100, 300))
            info_img = draw_text_to_image(info_img,
                                          "act. trans=%.3f %.3f" % (trans_actual_traj[counter][1],
                                                                    trans_actual_traj[counter][2]),
                                          (100, 400))
            info_img = draw_text_to_image(info_img, "ab rot=%.3f (%.3f)" % (ab_rot, ab_rot-90), (100, 500))
            info_img = draw_text_to_image(info_img, "dist1 =%.2f %.2f" % (ab_trans_image_space_x[counter], ab_trans_image_space_y[counter]), (100, 600))
            if head_rotations_by_masks[counter] is not None:
                info_img = draw_text_to_image(info_img, "rot. (head mask) =%.2f" % (-head_rotations_by_masks[counter]+90), (100, 900))

            if all_angles_before_null[counter] is None:
                info_img = draw_text_to_image(info_img, "missing comp.", (100, 800))

            if ab_centers[counter] is not None:
                dist = head_centers[counter] - ab_centers[counter]
                info_img = draw_text_to_image(info_img, "dist2 =%.2f %.2f" % (dist[1], dist[2]), (100, 700))
            res_im = np.vstack([np.hstack([seg_im1, im1]), np.hstack([seg_im2, info_img])])
            res_im = cv2.resize(res_im, (res_im.shape[1]//2, res_im.shape[0]//2))
            cv2.imwrite(f"test2/res-{ind}.png", res_im)


if __name__ == '__main__':
    visualize()
