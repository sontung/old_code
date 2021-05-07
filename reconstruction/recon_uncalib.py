import cv2
import numpy as np
import scipy
import sys
import random
import open3d as o3d
import pickle
import math
from tqdm import tqdm
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

x_total = []
y_total = []
for identifier in tqdm(lines, desc="Performing triangulation"):
    pairs = read_correspondence_from_dump("%s/dense-corr-%s.txt" % (dense_corr_dir, identifier))
    p1_t = []
    p2_t = []
    for x, y, x2, y2 in pairs:
        p1_t.append((x, y))
        p2_t.append((x2, y2))

    i = -1
    ep_vec = np.zeros((3, 0))
    f_mat = None
    while ep_vec.shape == (3, 0):
        f_mat, _ = cv2.findFundamentalMat(np.int32(p1_t[:i]), np.int32(p2_t[:i]), cv2.FM_RANSAC)
        f_mat = f_mat[:3, :]
        ep_vec = scipy.linalg.null_space(f_mat).squeeze()
        i -= 1

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

    colors = []
    xs2 = []
    ys2 = []

    for (x1, y1, x2, y2) in pairs:
        x_world, y_world, _, z_world = solver(full_proj2, full_proj1, y1, x1, y2, x2)
        xs2.append(x_world)
        ys2.append(y_world)

    xs = []
    ys = []
    x_mean = np.mean(xs2)
    y_mean = np.mean(ys2)
    with open("%s/point_cloud-%s.txt" % (saved_pc_dir, identifier), "w") as a_file:
        for (x1, y1, x2, y2) in pairs:
            x_world, y_world, _, z_world = solver(full_proj2, full_proj1, y1, x1, y2, x2)
            x_world -= x_mean
            y_world -= y_mean
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            color = img2[x2, y2]/255
            colors.append(color)
            print(x_world-np.mean(xs2), y_world-np.mean(ys2), z_world, color[2], color[1], color[0], file=a_file)

            xs.append(x_world)
            ys.append(y_world)
    x_total.append(np.mean(xs))
    y_total.append(np.mean(ys))
    # pcd = o3d.io.read_point_cloud("%s/point_cloud-%s.txt" % (saved_pc_dir, identifier), "xyzrgb")
    # o3d.visualization.draw_geometries([pcd])
    # break
print(x_total)
print(y_total)
plt.plot(x_total, y_total)
plt.show()
