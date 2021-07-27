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


def check_translation_bound(head_traj, ab_transx, ab_transy):
    os.makedirs("../data_heavy/area_compute/", exist_ok=True)
    pcd = o3d.io.read_triangle_mesh("../data/max-planck.obj")
    ab = o3d.io.read_triangle_mesh("../data/max-planck.obj")

    start_ab, _ = compute_ab_frames()
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.get_view_control().set_zoom(1.5)
    ab.translate([0, -ab_transx, -ab_transy])
    head_x_pos = []
    head_y_pos = []

    for counter in range(len(head_traj)):

        pcd.translate(head_traj[counter % len(head_traj)])

        vis.update_geometry(pcd)

        vis.poll_events()
        vis.update_renderer()
        head_x_pos.append(pcd.get_center()[1])
        head_y_pos.append(pcd.get_center()[2])

        if pcd.get_center()[1] < -ab_transx-5:
            correct = -ab_transx-5-pcd.get_center()[1]
            pcd.translate(-head_traj[counter % len(head_traj)])
            head_traj[counter % len(head_traj)][1] += correct
            pcd.translate(head_traj[counter % len(head_traj)])

        if pcd.get_center()[2] < -ab_transx-3:
            correct = -ab_transy-5-pcd.get_center()[2]
            pcd.translate(-head_traj[counter % len(head_traj)])
            head_traj[counter % len(head_traj)][2] += correct
            pcd.translate(head_traj[counter % len(head_traj)])

    vis.destroy_window()
    return head_traj


