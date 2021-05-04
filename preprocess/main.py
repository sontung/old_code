from os import listdir
from os.path import isfile, join
from edge_detection import process as edge_process_func
from utils import k_means_smoothing as smoothing_func
from tqdm import tqdm
import cv2
import numpy as np


def edge_detection():
    mypath = "../data_heavy/frames_ear_only"
    frames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    edge_saving_dir = "../data_heavy/frames_ear_only_with_edges"
    for im_name in tqdm(frames, desc="Extracting edges"):
        img = cv2.imread(join(mypath, im_name))
        img = smoothing_func(img)
        src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dst, detected_edges = edge_process_func(src_gray)
        cv2.imwrite("%s/%s" % (edge_saving_dir, im_name), np.hstack([dst, detected_edges]))


if __name__ == '__main__':
    edge_detection()
