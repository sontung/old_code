import os
from os import listdir
from os.path import isfile, join
from edge_detection import process as edge_process_func
from pp_utils import k_means_smoothing as smoothing_func
from pp_utils import kmeans_mask as smoothing_func2
from tqdm import tqdm
from glob import glob
from cpd import process_cpd, process_cpd_fast
import cv2
import pickle
import numpy as np


def edge_detection():
    mypath = "../data_heavy/frames_ear_only"
    frames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    edge_saving_dir = "../data_heavy/frames_ear_only_with_edges"
    pixels_path = "../data_heavy/frames_ear_coord_only"
    transform_path = "../data_heavy/edge_pixels"
    os.makedirs(edge_saving_dir, exist_ok=True)
    os.makedirs(pixels_path, exist_ok=True)
    os.makedirs(transform_path, exist_ok=True)

    for im_name in tqdm(frames, desc="Extracting edges"):

        with open(join(pixels_path, im_name), "rb") as fp:
            pixels_list = pickle.load(fp)
        if len(pixels_list) == 0:
            continue
        (xmax, ymax), (xmin, ymin) = np.max(pixels_list, axis=0)+3, np.min(pixels_list, axis=0)-3
        img_ori = cv2.imread(join(mypath, im_name))
        img_ori = smoothing_func2(pixels_list, img_ori)
        img = img_ori[xmin: xmax, ymin: ymax]

        src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, detected_edges = edge_process_func(src_gray)
        res = np.zeros((img_ori.shape[0], img_ori.shape[1]))
        res[xmin: xmax, ymin: ymax] = detected_edges

        nonzero_indices = np.nonzero(detected_edges)
        with open("%s/%s" % (transform_path, im_name), "w") as fp:
            for i in range(nonzero_indices[0].shape[0]):
                print(nonzero_indices[0][i],
                      nonzero_indices[1][i], file=fp)

        cv2.imwrite("%s/%s" % (edge_saving_dir, im_name), res)


def prepare_pixels_set():
    """
    prepare sets of pixels to match based on segmentation and edges
    """
    pixels_path = "../data_heavy/frames_ear_coord_only"
    edges_path = "../data_heavy/frames_ear_only_with_edges"
    saved_dir = "../data_heavy/refined_pixels"
    names = [f for f in listdir(pixels_path) if isfile(join(pixels_path, f))]
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    os.makedirs(saved_dir, exist_ok=True)

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


def simple_preprocess():
    """
    convert black background to non-black
    """
    mypath = "../data_heavy/frames_ear_only"
    frames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    saving_dir = "../data_heavy/frames_ear_only_nonblack_bg"
    pixels_path = "../data_heavy/frames_ear_coord_only"
    os.makedirs(saving_dir, exist_ok=True)

    for im_name in tqdm(frames, desc="Convert to non-black background"):
        img = cv2.imread(join(mypath, im_name))
        with open(join(pixels_path, im_name), "rb") as fp:
            pixels_list = pickle.load(fp)

        im_filled = np.zeros_like(img)

        for p in pixels_list:
            i, j = p
            im_filled[i, j] = 255

        indices = np.argwhere(im_filled[:, :, 0] == 0)
        img[indices[:, 0], indices[:, 1]] = (128, 128, 255)
        # for i, j in np.argwhere(im_filled[:, :, 0] == 0):
        #     img[i, j] = (128, 128, 255)
        # for i in range(img.shape[0]):
        #     for j in range(img.shape[1]):
        #         if im_filled[i, j, 0] == 0:
        #             img[i, j] = (128, 128, 255)
        cv2.imwrite("%s/%s" % (saving_dir, im_name), img)


def parametric_ellipse(rcenter, rw, rh, angle, p):
    x, y = p
    h, k = rcenter
    rx = rw/2
    ry = rh/2
    angle = np.radians(angle)

    t1 = (x - h)*np.cos(angle) + (y - k)*np.sin(angle)
    t2 = (x - h)*np.sin(angle) - (y - k)*np.cos(angle)

    dis = (t1*t1)/(rx*rx) + (t2*t2)/(ry*ry) - 1
    return np.abs(dis)


def fit_ellipse():
    saved_dir = "../data_heavy/line_images"
    os.makedirs(saved_dir, exist_ok=True)
    for afile in tqdm(glob('../data_heavy/head_rotations/*.png'), desc="Removing ellipse pixels"):
        image = cv2.imread(afile)
        indices = np.nonzero(image[:,:,0])
        nz_points = np.transpose((indices[1], indices[0])).astype(int)
        contours = nz_points.reshape((nz_points.shape[0], 1, nz_points.shape[1]))
        ellipse = cv2.fitEllipse(contours)

        center = ellipse[0]
        w, h = ellipse[1]
        angle = ellipse[2]

        bias = 0.2
        for p in nz_points:
            dis = parametric_ellipse(center, w, h, angle, p)
            if dis <= bias:
                image[p[1], p[0]] = (0, 0, 0)
        imn = afile.split("/")[-1]
        cv2.imwrite("%s/%s" % (saved_dir, imn), image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 50  # minimum number of pixels making up a line
        max_line_gap = 30  # maximum gap in pixels between connectable line segments

        lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

        angles = []
        angles_true = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 5)
                rot_deg = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
                angles.append(abs(rot_deg))
                angles_true.append(rot_deg)

        # cv2.imshow("t", image)
        # cv2.waitKey()
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    # extract_frame()
    edge_detection()
    process_cpd_fast()
    fit_ellipse()
    # prepare_pixels_set()
    # simple_preprocess()
