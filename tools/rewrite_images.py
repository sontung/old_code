import os
import open3d as o3d
import argparse
import pickle
import cv2
import sys
import numpy as np
from scipy.spatial.transform import Rotation as rot_mat_compute

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', type=str, default=False, help='directory', required=True)

args = vars(parser.parse_args())
RESULTS_DIR = args['dir']
COMPUTATIONS_DIR = f"{RESULTS_DIR}/everything_you_need.pkl"
if RESULTS_DIR.split('/')[-1]:
    VIDEO_NAME = RESULTS_DIR.split('/')[-1]
else:
    VIDEO_NAME = RESULTS_DIR.split('/')[-2]

NEW_DIR = f"data_video/all_final_vis/{VIDEO_NAME}2"
os.makedirs(NEW_DIR, exist_ok=True)


def new_model(debugging=False):
    texture = cv2.imread("data/model/textures/Head_albedo.jpg")
    texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)

    pcd_old = o3d.io.read_triangle_mesh("data/max-planck.obj")
    pcd_old.compute_vertex_normals()

    pcd = o3d.io.read_triangle_mesh("data/model/model.obj")
    pcd.compute_vertex_normals()

    triangle_uvs = np.asarray(pcd.triangle_uvs)
    triangle_uvs[:, 1] = 1 - triangle_uvs[:, 1]

    pcd.textures = [o3d.geometry.Image(texture)]

    # scale new_model to old_model
    area_scale = 980
    pcd.scale(area_scale, pcd.get_center())

    # rotation new model
    rot_mat = rot_mat_compute.from_euler('y', -180, degrees=True).as_matrix()
    pcd.rotate(rot_mat, pcd.get_center())
    pcd.translate(pcd_old.get_center(), relative=False)

    if debugging:

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_view_control().set_zoom(1.5)

        vis.add_geometry(pcd)
        vis.add_geometry(pcd_old)

        t = 0
        while t < 200:
            vis.poll_events()
            vis.update_renderer()
            t += 1

    return pcd


def modify_computed_data():
    with open(COMPUTATIONS_DIR, "rb") as pickle_file:
        all_computations = pickle.load(pickle_file)

    global_head_scale, global_ab_scale, trajectory, rotated_trajectory, \
    ab_transx, ab_transy, ab_rot, start_ab = all_computations

    global_ab_scale *= 1.5  # ab bigger
    for traj in trajectory:  # head goes further (x-axis)
        traj[2] -= 0.2

    # rewrite pick file
    results_to_disk = [global_head_scale, global_ab_scale,
                       trajectory, rotated_trajectory,
                       ab_transx, ab_transy, ab_rot,
                       start_ab]
    with open(f"{NEW_DIR}/everything_you_need.pkl", "wb") as pickle_file:
        pickle.dump(results_to_disk, pickle_file)

    return results_to_disk


def make_trajectory(values):
    res = []
    for i in range(1, len(values)):
        res.append(values[i]-values[i-1])
    return res


def make_rotations(size):
    roty = [0]*size
    roty[39] = 1
    roty[40] = 1
    roty[41] = 1
    roty[42] = 1
    roty[43] = 1
    roty[44] = 1
    roty[45] = 1
    roty[46] = 1
    roty[47] = 1
    roty[48] = 1
    roty[49] = 1
    roty[50] = 10
    roty[51] = 10
    roty[52] = 10
    roty[53] = 10
    roty[54] = 9
    roty[55] = 6
    roty[56] = 5
    roty[57] = 4
    roty[58] = 3
    roty[59] = 2
    roty[60] = 1
    roty[61] = 1
    roty[62] = 1
    roty[63] = 1
    roty[64] = 1
    roty[65] = 1
    roty[66] = 1
    roty[67] = 1
    roty[68] = 1
    roty[69] = 1



    rotz = [0]*size
    return make_trajectory(roty), make_trajectory(rotz)


def visualize(data, pcd):
    ab_mesh_dir = f"data_video/all_final_vis/{VIDEO_NAME}/mc_solutions_smoothed"
    saved_vis_dir = f"{NEW_DIR}/saved"
    os.makedirs(saved_vis_dir, exist_ok=True)
    pcd.translate([0, 0, 0], relative=False)
    pcd.compute_vertex_normals()

    ab_counter = 0
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.get_view_control().set_zoom(1.5)

    # write computations to disk
    global_head_scale, global_ab_scale, trajectory, rotated_trajectory, \
    ab_transx, ab_transy, ab_rot, start_ab = data

    rotated_trajectory_y,  rotated_trajectory_z = make_rotations(len(rotated_trajectory))

    # start writing visualizations
    trans_actual_traj = []
    rot_actual_traj = []
    pcd.scale(global_head_scale, pcd.get_center())
    head_centers = []
    ab_centers = []

    for counter in range(len(trajectory)):
        ab_added = False
        pcd.translate(trajectory[counter % len(trajectory)])
        rot_mat = rot_mat_compute.from_euler('x', rotated_trajectory[counter % len(trajectory)],
                                             degrees=True).as_matrix()
        pcd.rotate(rot_mat, pcd.get_center())
        trans_actual_traj.append(trajectory[counter % len(trajectory)])
        rot_actual_traj.append(rotated_trajectory[counter % len(trajectory)])

        rot_mat_z = rot_mat_compute.from_euler('z', -rotated_trajectory_z[counter % len(trajectory)],
                                               degrees=True).as_matrix()
        pcd.rotate(rot_mat_z, pcd.get_center())
        rot_mat_y = rot_mat_compute.from_euler('y', -rotated_trajectory_y[counter % len(trajectory)],
                                               degrees=True).as_matrix()
        pcd.rotate(rot_mat_y, pcd.get_center())

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
        vis.capture_screen_image(f"{saved_vis_dir}/v1-%s.png" % counter, do_render=True)

        vis.get_view_control().rotate(500, 0)
        vis.capture_screen_image(f"{saved_vis_dir}/v2-%s.png" % counter, do_render=True)
        if ab_added and ab is not None:
            vis.remove_geometry(ab, reset_bounding_box=False)
    vis.destroy_window()


def final_vis():
    save_dir = f"{NEW_DIR}/final_vis"
    os.makedirs(save_dir, exist_ok=True)
    sys.stdin = open("data_heavy/frames/info.txt")
    lines = [du[:-1] for du in sys.stdin.readlines()]
    lines = lines[1:]
    recon_dir = f"{NEW_DIR}/saved"
    segment_dir = "data_heavy/frames_seg_abh"
    final_im_size = None

    for idx_recon, idx_seg in enumerate(lines):
        seg_im = cv2.imread("%s/1-%s.png" % (segment_dir, idx_seg))
        recon_im = cv2.imread("%s/v1-%s.png" % (recon_dir, idx_recon))
        recon_im2 = cv2.imread("%s/v2-%s.png" % (recon_dir, idx_recon))
        if final_im_size is None:
            final_im_size = (seg_im.shape[1], seg_im.shape[0])
        ori_im = cv2.imread("%s/1-%s.png" % ("data_heavy/frames", idx_seg))
        ori_im2 = cv2.imread("%s/2-%s.png" % ("data_heavy/frames", idx_seg))
        if np.any([du is None for du in [recon_im, recon_im2, ori_im, ori_im2, seg_im]]):
            break
        seg_im = cv2.resize(seg_im, final_im_size)
        ori_im = cv2.resize(ori_im, final_im_size)
        blend = cv2.addWeighted(ori_im, 0.3, seg_im, 0.7, 0)
        recon_im = cv2.resize(recon_im, final_im_size)
        recon_im2 = cv2.resize(recon_im2, final_im_size)

        ori_im2 = cv2.resize(ori_im2, final_im_size)
        seg_im2 = cv2.resize(cv2.imread("%s/2-%s.png" % (segment_dir, idx_seg)), final_im_size)
        blend2 = cv2.addWeighted(ori_im2, 0.3, seg_im2, 0.7, 0)
        final_im = np.hstack([blend2, blend])
        final_im2 = np.hstack([recon_im2, recon_im])
        all_im = np.vstack([final_im, final_im2])
        cv2.imwrite(f"{save_dir}/{idx_recon}.png", all_im)


if __name__ == '__main__':
    pcd_ = new_model()
    data = modify_computed_data()
    visualize(data, pcd_)
    final_vis()
