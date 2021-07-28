from functools import partial
import matplotlib.pyplot as plt
from pycpd import AffineRegistration
from fast_registration import register_fast
from numba import config, njit, threading_layer

import numpy as np
from tqdm import tqdm
import glob
import cv2
import os


def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], color='blue', label='Source')
    plt.text(0.87, 0.92, 'Iteration: {:d}'.format(
        iteration), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)
    plt.savefig(f"test/t{iteration}.png")


def normalize(inp, ref):
    inp[:, 0] = (np.max(ref[:, 0]) - np.min(ref[:, 0]))*(inp[:, 0] - np.min(inp[:, 0]))/(np.max(inp[:, 0]) - np.min(inp[:, 0])) + np.min(ref[:, 0])
    inp[:, 1] = (np.max(ref[:, 1]) - np.min(ref[:, 1]))*(inp[:, 1] - np.min(inp[:, 1]))/(np.max(inp[:, 1]) - np.min(inp[:, 1])) + np.min(ref[:, 1])
    return inp


def process_cpd_with_vis():
    ear = cv2.imread("/home/sontung/work/3d-air-bag-p2/data/ear.png")
    ear = cv2.resize(ear, (ear.shape[1]//4, ear.shape[0]//4))
    all_files = glob.glob("../data_heavy/edge_pixels/*")

    nonzero_indices = np.nonzero(ear)
    with open("../data/ear.txt", "w") as fp:
        for i in range(nonzero_indices[0].shape[0]):
            print(nonzero_indices[0][i],
                  nonzero_indices[1][i], file=fp)

    y_data = np.loadtxt('../data/ear.txt')
    y_data2 = np.loadtxt('../data/ear.txt')

    for file in all_files:
        if file.split("/")[-1] != "0-99.png":
            continue
        x_data = np.loadtxt(file)

        fig = plt.figure()
        fig.add_axes([0, 0, 1, 1])
        callback = partial(visualize, ax=fig.axes[0])

        reg = AffineRegistration(**{'X': x_data, 'Y': y_data}, max_iterations=100)
        reg.register()
        y_data = reg.transform_point_cloud(y_data)

        y_data_norm = normalize(y_data, x_data)
        reg = AffineRegistration(**{'X': x_data, 'Y': y_data_norm}, max_iterations=45)
        reg.register(callback)
        y_data_norm = reg.transform_point_cloud(y_data_norm)

        ax = fig.axes[0]
        ax.scatter(x_data[:, 0],  x_data[:, 1], color='red', label='Target')
        ax.scatter(y_data[:, 0],  y_data[:, 1], color='blue', label='Source')
        ax.scatter(y_data_norm[:, 0],  y_data_norm[:, 1], color='yellow', label='Source norm')

        ax.legend(loc='upper left', fontsize='x-large')
        plt.show()


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


def process_cpd(debug=False):
    ear = cv2.imread("../data/ear.png")
    ear = cv2.resize(ear, (ear.shape[1]//4, ear.shape[0]//4))
    all_files = glob.glob("../data_heavy/edge_pixels/*")
    transform_path = "../data_heavy/head_rotations"
    os.makedirs(transform_path, exist_ok=True)
    debug_path = "../data_heavy/head_rotations_debug"
    if debug:
        os.makedirs(debug_path, exist_ok=True)

    nonzero_indices = np.nonzero(ear)
    with open("../data/ear.txt", "w") as fp:
        for i in range(nonzero_indices[0].shape[0]):
            print(nonzero_indices[0][i],
                  nonzero_indices[1][i], file=fp)

    y_data = np.loadtxt('../data/ear.txt')

    for file in tqdm(all_files, "Extracting rotation using CPD"):

        x_data = np.loadtxt(file)
        reg = AffineRegistration(**{'X': x_data, 'Y': y_data}, max_iterations=45)
        reg.register()
        y_data_transformed = reg.transform_point_cloud(y_data)
        imn = file.split("/")[-1]
        cv2.imwrite(f"{transform_path}/{imn}", draw_image(y_data_transformed))

        if debug:
            cv2.imwrite(f"{debug_path}/{imn}-res.png", draw_image(y_data_transformed))
            cv2.imwrite(f"{debug_path}/{imn}-inp.png", draw_image(x_data))


def process_cpd_fast(debug=False):

    ear = cv2.imread("../data/ear.png")
    ear = cv2.resize(ear, (ear.shape[1]//4, ear.shape[0]//4))
    all_files = glob.glob("../data_heavy/edge_pixels/*")
    transform_path = "../data_heavy/head_rotations"
    os.makedirs(transform_path, exist_ok=True)
    debug_path = "../data_heavy/head_rotations_debug"
    if debug:
        os.makedirs(debug_path, exist_ok=True)

    nonzero_indices = np.nonzero(ear)
    with open("../data/ear.txt", "w") as fp:
        for i in range(nonzero_indices[0].shape[0]):
            print(nonzero_indices[0][i],
                  nonzero_indices[1][i], file=fp)

    y_data = np.loadtxt('../data/ear.txt')
    for afile in tqdm(all_files, desc="Extracting rotation using affine CPD"):
        imn = afile.split("/")[-1]
        if debug and imn != "1-48.png":
            continue
        x_data = np.loadtxt(afile)
        y_data_norm = normalize(y_data, x_data)
        y_data_transformed, b, t, error = register_fast(x_data, y_data_norm)
        if debug:
            fig = plt.figure()
            fig.add_axes([0, 0, 1, 1])
            ax = fig.axes[0]
            ax.scatter(x_data[:, 0],  x_data[:, 1], color='red', label='Target')
            ax.scatter(y_data_norm[:, 0],  y_data_norm[:, 1], color='blue', label='Source')
            ax.scatter(y_data_transformed[:, 0],  y_data_transformed[:, 1], color='yellow', label='Source norm')
            ax.legend(loc='upper left', fontsize='x-large')
            plt.show()

        cv2.imwrite(f"{transform_path}/{imn}", draw_image(y_data_transformed))

        # cv2.imshow("t", draw_image(y_data_transformed))
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        if debug:
            cv2.imwrite(f"{debug_path}/{imn}-res.png", draw_image(y_data_transformed))
            cv2.imwrite(f"{debug_path}/{imn}-inp.png", draw_image(x_data))


if __name__ == '__main__':
    process_cpd_fast(True)
    # process_cpd_with_vis()
