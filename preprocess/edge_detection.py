import cv2 as cv
import argparse
from utils import k_means_smoothing as smoothing_func

from os import listdir
from os.path import isfile, join


def demo():
    max_lowThreshold = 150
    window_name = 'Edge Map'
    title_trackbar = 'Min Threshold:'
    ratio = 3
    kernel_size = 1

    mypath = "../data_heavy/frames_ear_only"
    frames = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    def CannyThreshold(val):
        low_threshold = val
        img_blur = cv.blur(src_gray, (kernel_size,kernel_size))
        detected_edges = cv.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size, L2gradient=True)
        print(low_threshold, low_threshold*ratio)
        mask = detected_edges != 0
        dst = src * (mask[:,:,None].astype(src.dtype))
        cv.imshow(window_name, dst)

    parser = argparse.ArgumentParser(description='Code for Canny Edge Detector tutorial.')
    parser.add_argument('--input', help='Path to input image.', default='0')
    args = parser.parse_args()
    src = cv.imread(join(mypath, frames[int(args.input)]))
    src = smoothing_func(src)
    src = cv.resize(src, (src.shape[1]//4, src.shape[0]//4))

    if src is None:
        print('Could not open or find the image: ', args.input)
        exit(0)
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    cv.namedWindow(window_name)
    cv.createTrackbar(title_trackbar, window_name, 0, max_lowThreshold, CannyThreshold)
    CannyThreshold(0)
    cv.waitKey()


def process(gray_image, low_threshold=0, ratio=3, kernel_size=1):
    detected_edges = cv.Canny(gray_image, low_threshold, low_threshold * ratio, kernel_size)
    mask = detected_edges != 0
    dst = gray_image * (mask.astype(gray_image.dtype))
    return dst, detected_edges


if __name__ == '__main__':
    demo()
