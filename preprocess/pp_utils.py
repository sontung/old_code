import os

import cv2
import numpy as np
import os
import sys
from os import listdir
from os.path import isfile, join
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.datasets import load_sample_image
from time import time
from pathlib import Path
from tqdm import tqdm


def keep_one_biggest_contour(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cnts, _ = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = None
    if len(cnts) > 0:
        largest_cnt = max(cnts, key=lambda du1: cv2.contourArea(du1))
        res = np.zeros_like(img) 
        mask = np.zeros([img.shape[0], img.shape[1], 3])
        cv2.fillPoly(res, pts=[largest_cnt], color=(192, 128, 128))
        img[res != (192, 128, 128)] = 0
        mask[res == (192, 128, 128)] = 1
        mask = mask[:, :, 0]
        #cv2.imshow("t", np.hstack([res, img]))
        #cv2.waitKey()
        #cv2.destroyAllWindows()
    return img, mask


def extract_frame(mypath='../data_const/run', output='../data_heavy/frames'):
    """
    extract frames from videos stored in ../data_heavy/run
    """
    # mypath = "../data_const/run"
    videos = sorted([join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))])
    print("Extracting frames from", videos)
    # os.makedirs("../data_heavy/frames", exist_ok=True)
    os.makedirs(output, exist_ok=True)

    min_nb_frame = None
    keep_counts = []
    for c, v in enumerate(videos):
        cap = cv2.VideoCapture(v)
        count = 0
        frame_idx = 0
        counts = []

        while True:
            ret, frame = cap.read()
            if ret:
                count += 1
                if count % 3 == 0:
                    frame_idx += 1
                    scale = np.min(frame.shape[:2]) // 500
                    frame = cv2.resize(frame, (frame.shape[1]//scale, frame.shape[0]//scale))
                    cv2.imwrite(f'{output}/%d-%d.png' % (c, frame_idx), frame)
                    counts.append(frame_idx)
            else:
                break
        if min_nb_frame:
            if min_nb_frame > count:
                keep_counts = counts
                min_nb_frame = count
        else:
            keep_counts = counts
            min_nb_frame = count
    keep_counts = sorted(keep_counts)
    with open(join(output, "info.txt"), "w") as text_file:
        for c in keep_counts:
            print(c, file=text_file)


def kmeans_mask(pixels, image):
    n_colors = 2
    image_array = np.zeros((pixels.shape[0], 3), dtype=np.float)
    for i, (u, v) in enumerate(pixels):
        image_array[i] = image[u, v]/255.0
    image_array_sample = shuffle(image_array, random_state=0)
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

    # Get labels for all points
    labels = kmeans.predict(image_array)
    label2color = {0: np.array([0.5, 0.6, 0.7]),
                   1: np.array([0.1, 0.2, 0.3]),
                   2: np.array([0.4, 0.7, 0.3])}
    for i, (u, v) in enumerate(pixels):
        image[u, v] = label2color[labels[i]]*255
    return image
    # cv2.imshow("t", image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # sys.exit()


def k_means_smoothing(rgb):
    """
    perform k means segmentation to smooth
    """
    n_colors = 2

    # Load the Summer Palace photo
    china = rgb
    # china = load_sample_image('flower.jpg')

    # Convert to floats instead of the default 8 bits integer coding. Dividing by
    # 255 is important so that plt.imshow behaves works well on float data (need to
    # be in the range [0-1])
    china = np.array(china, dtype=np.float64) / 255

    # Load Image and transform to a 2D numpy array.
    w, h, d = tuple(china.shape)
    assert d == 3
    image_array = np.reshape(china, (w * h, d))

    image_array_sample = shuffle(image_array, random_state=0)
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

    # Get labels for all points
    labels = kmeans.predict(image_array)

    def recreate_image(labels, w, h, codebook, d):
        """Recreate the (compressed) image from the code book & labels"""
        image = np.zeros((w, h, d))
        label_idx = 0
        for i in range(w):
            for j in range(h):
                image[i][j] = codebook[labels[label_idx]]*255
                label_idx += 1
        return image.astype(np.uint8)

    res = recreate_image(labels, w, h, {0: np.array([0.5, 0.6, 0.7]),
                                        1: np.array([0.1, 0.2, 0.3]),
                                        2: np.array([0.4, 0.7, 0.3])},
                         kmeans.cluster_centers_.shape[1])
    return res
    # from edge_detection import process
    # res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # edge1, edge2 = process(res)
    # cv2.imshow("test", np.hstack([edge1, edge2, res]))
    # cv2.waitKey()
    # cv2.destroyAllWindows()

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


def extract_all_video(root='/media/hblab/01D5F2DD5173DEA0/AirBag/3d-air-bag-p2/data_video'):
    videos = 'all_video'
    frames = 'all_frames'

    sub_videos = os.walk(os.path.join(root, videos)).__next__()[1]
    sub_frames = os.walk(os.path.join(root, frames)).__next__()[1]
    for v in tqdm(sub_videos, desc="Extracting video"):
        if v in sub_frames:
            print(f'Skipping {v}')
        else:
            extract_frame(mypath=os.path.join(root, videos, v), output=os.path.join(root, frames, v))
    return


if __name__ == '__main__':
    extract_frame()
    # extract_all_video()
    # k_means_smoothing(None)
