from os import listdir
from os.path import isfile, join
from edge_detection import process as edge_process_func
from utils import k_means_smoothing as smoothing_func
from tqdm import tqdm
import cv2
import pickle
import numpy as np


def edge_detection():
    mypath = "../data_heavy/frames_ear_only"
    frames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    edge_saving_dir = "../data_heavy/frames_ear_only_with_edges"
    for im_name in tqdm(frames, desc="Extracting edges"):
        img = cv2.imread(join(mypath, im_name))
        img = smoothing_func(img)
        src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, detected_edges = edge_process_func(src_gray)
        cv2.imwrite("%s/%s" % (edge_saving_dir, im_name), detected_edges)


def prepare_pixels_set():
    """
    prepare sets of pixels to match based on segmentation and edges
    """
    pixels_path = "../data_heavy/frames_ear_coord_only"
    edges_path = "../data_heavy/frames_ear_only_with_edges"
    saved_dir = "../data_heavy/refined_pixels"
    names = [f for f in listdir(pixels_path) if isfile(join(pixels_path, f))]
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    for im_name in tqdm(names, desc="Extracting refined pixel list for matching"):
        edge_im = cv2.imread(join(edges_path, im_name), 0)
        edge_im_filled = np.zeros_like(edge_im)

        refined_pixel_list = []
        with open(join(pixels_path, im_name), "rb") as fp:
            pixels_list = pickle.load(fp)

        for p in pixels_list:
            i, j = p
            edge_im_filled[i, j] = 255

        for p in pixels_list:
            i, j = p
            if edge_im[i, j] > 0:
                refined_pixel_list.append(p)
            else:
                for u, v in neighbors:
                    if edge_im_filled[i+u, j+v] == 0:
                        refined_pixel_list.append(p)
                        break

        # test
        # edge_im_filled = np.zeros_like(edge_im)
        # for p in refined_pixel_list:
        #     i, j = p
        #     edge_im_filled[i, j] = 255

        with open(join(saved_dir, im_name), "wb") as fp:
            pickle.dump(refined_pixel_list, fp)


if __name__ == '__main__':
    # edge_detection()
    prepare_pixels_set()
