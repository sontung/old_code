import cv2
import numpy as np

from os import listdir
from os.path import isfile, join
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from time import time


def extract_frame():
    """
    extract frames from videos stored in ../data_heavy/run
    """
    mypath = "../data_heavy/run"
    videos = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    print("Extracting frames from", videos)
    for c, v in enumerate(videos):
        cap = cv2.VideoCapture(v)
        count = 0

        while True:
            ret, frame = cap.read()
            if ret:
                count += 1
                if count % 3 == 0:
                    cv2.imwrite('../data_heavy/frames/%d-%d.png' % (c, count), frame)
                if count > 100:
                    break


def k_means_smoothing(rgb):
    n_colors = 2

    # Load the Summer Palace photo
    china = rgb

    # Convert to floats instead of the default 8 bits integer coding. Dividing by
    # 255 is important so that plt.imshow behaves works well on float data (need to
    # be in the range [0-1])
    china = np.array(china, dtype=np.float64) / 255

    # Load Image and transform to a 2D numpy array.
    w, h, d = original_shape = tuple(china.shape)
    assert d == 3
    image_array = np.reshape(china, (w * h, d))

    image_array_sample = shuffle(image_array, random_state=0)
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

    # Get labels for all points
    labels = kmeans.predict(image_array)

    def recreate_image(codebook, labels, w, h):
        """Recreate the (compressed) image from the code book & labels"""
        d = codebook.shape[1]
        image = np.zeros((w, h, d))
        label_idx = 0
        for i in range(w):
            for j in range(h):
                image[i][j] = codebook[labels[label_idx]]*255
                label_idx += 1
        return image.astype(np.uint8)

    return recreate_image(kmeans.cluster_centers_, labels, w, h)


if __name__ == '__main__':
    extract_frame()
