import cv2
import numpy as np
import scipy
import sys
import random
import open3d as o3d
import pickle
import math
from rec_utils import read_correspondence_from_dump
from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path


def solver(p1, p2, u1, v1, u2, v2):
    mat1 = np.asarray([
        (u1 * p1[2, :] - p1[0, :]),
        (v1 * p1[2, :] - p1[1, :]),
        (u2 * p2[2, :] - p2[0, :]),
        (v2 * p2[2, :] - p2[1, :])
    ])

    _, _, V = np.linalg.svd(mat1)
    res = V[-1, :4]
    res = res / res[3]
    return res


sys.stdin = open("../data_heavy/frames/info.txt")
lines = [du[:-1] for du in sys.stdin.readlines()]
dense_corr_dir = "../data_heavy/matching_solutions"
images_dir = "../data_heavy/frames_ear_only_nonblack_bg"
saved_pc_dir = "../data_heavy/point_cloud_solutions"
for identifier in lines:
    pairs = read_correspondence_from_dump("%s/dense-corr-%s.txt" % (dense_corr_dir, identifier))
    p1_t = []
    p2_t = []
    for x, y, x2, y2 in pairs:
        p1_t.append((x, y))
        p2_t.append((x2, y2))

    f_mat, _ = cv2.findFundamentalMat(np.int32(p1_t), np.int32(p2_t), cv2.FM_RANSAC)
    f_mat = f_mat[:3, :]
    ep_vec = scipy.linalg.null_space(f_mat).squeeze()

    a1, a2, a3 = ep_vec
    skew_mat = np.array([[0, -a3, a2],
                         [a3, 0, -a1],
                         [-a2, a1, 0]])
    skew_mat2 = np.matmul(skew_mat, f_mat)
    full_proj2 = np.hstack([skew_mat2, np.expand_dims(ep_vec, 1)])
    full_proj2 = np.vstack([full_proj2, np.array([0, 0, 0, 1])])
    full_proj1 = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]]).astype(np.float)

    full_proj1[:, 2] = full_proj2[:, 2]
    full_proj1[:, 3] = full_proj2[:, 3]
    full_proj2[:, 2] = np.array([0, 0, 1, 0])
    full_proj2[:, 3] = np.array([0, 0, 0, 1])

    img1 = cv2.imread("%s/0-%s.png" % (images_dir, identifier))
    img2 = cv2.imread("%s/1-%s.png" % (images_dir, identifier))

    # correspondences
    LOAD_PRE_COMPUTED_CORR = True

    X = []
    Y = []
    Z = []
    colors = []
    with open("%s/point_cloud-%s.txt" % (saved_pc_dir, identifier), "w") as a_file:
        for (x1, y1, x2, y2) in pairs:
            x_world, y_world, _, z_world = solver(full_proj2, full_proj1, y2, x2, y1, x1)
            X.append(x_world)
            Y.append(y_world)
            Z.append(z_world)
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            color = img1[x1, y1]/255
            colors.append(color)
            print(x_world, y_world, z_world, color[2], color[1], color[0], file=a_file)

    pcd = o3d.io.read_point_cloud("%s/point_cloud-%s.txt" % (saved_pc_dir, identifier), "xyzrgb")
    o3d.visualization.draw_geometries([pcd])

