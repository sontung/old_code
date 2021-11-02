import glob
import math
import os
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as rot_mat_compute
from tqdm import tqdm
from test_model import new_model, new_model_no_normal


def sim_prior_view(trajectory, rotated_trajectory, ab_transx2, ab_transy2, ab_comp,
                   head_scaling_additional=None, scale_both=None):
    os.makedirs("../data_heavy/area_compute/", exist_ok=True)
    pcd = new_model_no_normal()
    pcd.translate([0, 0, 0], relative=False)
    ab_scale, ab_rot, ab_area, head_area = ab_comp

    # this is just some inaccurate scaling for getting the sizes of both head and ab equal
    global_scale_ab_list = []
    ab = o3d.io.read_triangle_mesh("../new_particles_63.obj")
    scale1 = pcd.get_surface_area() / ab.get_surface_area()
    global_scale_ab_list.append(math.sqrt(scale1 / ab_scale))
    global_scale_ab = np.mean(global_scale_ab_list)
    if scale_both:
        global_scale_ab *= scale_both

    ab_counter = 0
    start_ab = 0
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.get_view_control().set_zoom(1.5)

    if head_scaling_additional is not None:
        pcd.scale(head_scaling_additional*scale_both, pcd.get_center())

    for counter in range(len(trajectory)):
        ab_added = False

        rot1 = rotated_trajectory[0]
        if rot1 > 90:
            rot_mat2 = rot_mat_compute.from_euler('x', -rot1+90,
                                                  degrees=True).as_matrix()
        else:
            rot_mat2 = rot_mat_compute.from_euler('x', rot1-90,
                                                  degrees=True).as_matrix()

        pcd.rotate(rot_mat2, pcd.get_center())

        vis.update_geometry(pcd)

        vis.get_view_control().rotate(-500, 0)
        vis.capture_screen_image("head-%s.png" % counter, do_render=True)
        ab = None

        if counter >= start_ab-1:
            ab = o3d.io.read_triangle_mesh(f"../new_particles_63.obj")
            ab.scale(global_scale_ab, ab.get_center())
            ab.translate([0, ab_transx2, ab_transy2])
            ab.rotate(rot_mat_compute.from_euler("y", 90, degrees=True).as_matrix())
            if ab_rot > 90:
                rot_mat2 = rot_mat_compute.from_euler('x', -ab_rot + 90,
                                                      degrees=True).as_matrix()
            else:
                rot_mat2 = rot_mat_compute.from_euler('x', ab_rot - 90,
                                                      degrees=True).as_matrix()
            ab.rotate(rot_mat2, ab.get_center())
            vis.remove_geometry(pcd, reset_bounding_box=False)
            vis.add_geometry(ab, reset_bounding_box=False)
            vis.capture_screen_image("ab-%s.png" % counter, do_render=True)
            vis.add_geometry(pcd, reset_bounding_box=False)
            ab_counter += 1

        if ab_added and ab is not None:
            vis.remove_geometry(ab, reset_bounding_box=False)
        vis.get_view_control().rotate(500, 0)
        vis.poll_events()
        vis.update_renderer()
    vis.destroy_window()

    image_names = glob.glob("head*")
    arr = []
    for name in image_names:
        im = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        arr.append(np.sum(im != 255))
    head_area = np.max(arr)

    image_names = glob.glob("ab*")
    arr = []
    for name in image_names:
        im = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        arr.append(np.sum(im != 255))
    ab_area = np.max(arr)
    return head_area, ab_area, global_scale_ab


def compute_head_ab_areas(comp):
    """
    prior simulation to compute head/ab scale ratio
    """

    ab_scale = comp["head pixels"]/comp["ab pixels"]*1.0
    ab_rot = comp["ab rot"]
    head_rot = comp["head rot"]
    ab_area = comp["ab pixels"]
    head_area = comp["head pixels"]
    ab_transy, ab_transx = comp["head center"] - comp["ab center"]

    head_area, ab_area, _ = sim_prior_view([[0, 0, 0]], [head_rot], ab_transx, ab_transy,
                                           (ab_scale, ab_rot, ab_area, head_area))

    scale1 = head_area/ab_area
    img_ab_area, img_head_area = ab_area, head_area

    if ab_scale >= 0.27:
        head_scale = np.sqrt(0.27/scale1)
    else:
        head_scale = np.sqrt(ab_scale/scale1)

    scale_both = np.sqrt((img_head_area/head_area+img_ab_area/ab_area)/2)
    head_area, ab_area, ab_scale_final = sim_prior_view([[0, 0, 0]], [head_rot], ab_transx, ab_transy,
                                                        (ab_scale, ab_rot, ab_area, head_area),
                                                        head_scaling_additional=head_scale, scale_both=scale_both)
    results = {"head scale": head_scale*scale_both,
               "ab scale": ab_scale_final,
               "trajectory": [[0, 0, 0]],
               "rot trajectory": [head_rot]}
    results2 = [ab_scale, ab_transx, ab_transy, ab_rot]
    return results, results2, (ab_transx, ab_transy)


def visualize_wo_ab(comp1, comp2):
    pcd = new_model()
    pcd.translate([0, 0, 0], relative=False)
    pcd.compute_vertex_normals()
    head_rot = comp1["head rot"]

    trajectory = [[0, 0, 0]]
    rotated_trajectory = [head_rot]

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.get_view_control().set_zoom(1.5)

    # write computations to disk
    # results_to_disk = [global_head_scale, global_ab_scale,
    #                    trajectory, rotated_trajectory,
    #                    ab_transx, ab_transy, ab_rot,
    #                    start_ab]
    # os.makedirs("../data_const/final_vis", exist_ok=True)
    # with open("../data_const/final_vis/everything_you_need.pkl", "wb") as pickle_file:
    #     pickle.dump(results_to_disk, pickle_file)

    # start writing visualizations
    for counter in tqdm(range(len(trajectory)), desc="Completing simulation"):
        rot1 = rotated_trajectory[0]
        if rot1 > 90:
            rot_mat2 = rot_mat_compute.from_euler('x', -rot1 + 90,
                                                  degrees=True).as_matrix()
        else:
            rot_mat2 = rot_mat_compute.from_euler('x', rot1 - 90,
                                                  degrees=True).as_matrix()

        rot2 = comp2["head rot"]
        rot_mat = rot_mat_compute.from_euler('z', rot2-90,
                                             degrees=True).as_matrix()
        pcd.rotate(rot_mat, pcd.get_center())

        pcd.rotate(rot_mat2, pcd.get_center())
        vis.update_geometry(pcd)

        vis.get_view_control().rotate(-500, 0)
        vis.capture_screen_image("v1.png", do_render=True)

        vis.get_view_control().rotate(500, 0)
        vis.capture_screen_image("v2.png", do_render=True)
    vis.destroy_window()
    return cv2.imread("v1.png"), cv2.imread("v2.png")


def visualize(comp1, comp2):
    pcd = new_model()
    pcd.translate([0, 0, 0], relative=False)
    pcd.compute_vertex_normals()
    du_outputs, du_outputs2, (_, _) = compute_head_ab_areas(comp1)
    _, ab_transx, ab_transy, ab_rot = du_outputs2

    trajectory = du_outputs["trajectory"]
    rotated_trajectory = du_outputs["rot trajectory"]
    global_head_scale = du_outputs["head scale"]
    global_ab_scale = du_outputs["ab scale"]

    start_ab = 0
    ab_counter = 0
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.get_view_control().set_zoom(1.5)

    # write computations to disk
    # results_to_disk = [global_head_scale, global_ab_scale,
    #                    trajectory, rotated_trajectory,
    #                    ab_transx, ab_transy, ab_rot,
    #                    start_ab]
    # os.makedirs("../data_const/final_vis", exist_ok=True)
    # with open("../data_const/final_vis/everything_you_need.pkl", "wb") as pickle_file:
    #     pickle.dump(results_to_disk, pickle_file)

    # start writing visualizations
    pcd.scale(global_head_scale, pcd.get_center())
    for counter in tqdm(range(len(trajectory)), desc="Completing simulation"):
        ab_added = False
        rot1 = rotated_trajectory[0]
        if rot1 > 90:
            rot_mat2 = rot_mat_compute.from_euler('x', -rot1+90,
                                                  degrees=True).as_matrix()
        else:
            rot_mat2 = rot_mat_compute.from_euler('x', rot1-90,
                                                  degrees=True).as_matrix()
        pcd.rotate(rot_mat2, pcd.get_center())

        rot2 = comp2["head rot"]
        rot_mat = rot_mat_compute.from_euler('z', rot2-90,
                                             degrees=True).as_matrix()
        pcd.rotate(rot_mat, pcd.get_center())
        vis.update_geometry(pcd)

        ab = None
        if counter >= start_ab-1:
            ab = o3d.io.read_triangle_mesh(f"../new_particles_63.obj")
            ab.translate([0, 0, 0], relative=False)
            ab.compute_vertex_normals()
            ab.scale(global_ab_scale, ab.get_center())
            ab.translate([0, ab_transx, ab_transy])
            ab.rotate(rot_mat_compute.from_euler("y", 90, degrees=True).as_matrix())

            if ab_rot > 90:
                rot_mat2 = rot_mat_compute.from_euler('x', -ab_rot + 90,
                                                      degrees=True).as_matrix()
            else:
                rot_mat2 = rot_mat_compute.from_euler('x', ab_rot - 90,
                                                      degrees=True).as_matrix()
            ab.rotate(rot_mat2, ab.get_center())

            rot2 = comp2["ab rot"]
            rot_mat = rot_mat_compute.from_euler('z', rot2 - 90,
                                                 degrees=True).as_matrix()
            ab.rotate(rot_mat, ab.get_center())
            
            trans = comp2["ab center"] - comp2["head center"]
            ab.translate([trans[1], trans[0], 0])

            vis.add_geometry(ab, reset_bounding_box=False)
            ab_counter += 1

        vis.get_view_control().rotate(-500, 0)
        vis.capture_screen_image("v1.png", do_render=True)

        vis.get_view_control().rotate(500, 0)
        vis.capture_screen_image("v2.png", do_render=True)
        if ab_added and ab is not None:
            vis.remove_geometry(ab, reset_bounding_box=False)
    vis.destroy_window()
    return cv2.imread("v1.png"), cv2.imread("v2.png")



if __name__ == '__main__':
    visualize()
