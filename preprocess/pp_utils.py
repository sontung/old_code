import os

import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.datasets import load_sample_image
from time import time
from pathlib import Path


def extract_frame():
    """
    extract frames from videos stored in ../data_heavy/run
    """
    mypath = "../data_const/run"
    videos = sorted([join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))])
    print("Extracting frames from", videos)
    counts = []
    os.makedirs("../data_heavy/frames", exist_ok=True)

    for c, v in enumerate(videos):
        cap = cv2.VideoCapture(v)
        count = 0

        while True:
            ret, frame = cap.read()
            if ret:
                count += 1
                if count % 3 == 0:
                    # Path("../data_heavy/sfm_data/%d/images" % count).mkdir(parents=True, exist_ok=True)
                    # cv2.imwrite("../data_heavy/sfm_data/%d/images/%d-%d.png" % (count, c, count), frame)
                    cv2.imwrite('../data_heavy/frames/%d-%d.png' % (c, count), frame)
                    counts.append(count)
            else:
                break

    counts = set(counts)
    with open(join("../data_heavy/frames", "info.txt"), "w") as text_file:
        for c in counts:
            print(c, file=text_file)


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
                                        1: np.array([0.1, 0.2, 0.3])},
                         kmeans.cluster_centers_.shape[1])
    return res
    # from edge_detection import process
    # res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # edge1, edge2 = process(res)
    # cv2.imshow("test", np.hstack([edge1, edge2, res]))
    # cv2.waitKey()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    extract_frame()
    # k_means_smoothing(None)
