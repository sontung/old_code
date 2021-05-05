import sys
import pickle
import cv2
from utils import compute_zncc


def match_zncc(im1, im2, pixels1, pixels2):
    corr = []
    for x, y in pixels1:
        a_match = None
        best_score = 0
        for x2, y2 in pixels2:
            score = compute_zncc(x, y, x2, y2, im1, im2, 17)
            if score > best_score:
                best_score = score
                a_match = (x2, y2)
        corr.append((x, y, a_match[0], a_match[1]))
    return corr


def main():
    sys.stdin = open("../data_heavy/frames/info.txt")
    lines = [du[:-1] for du in sys.stdin.readlines()]
    print(lines)
    refined_pixel_dir = "../data_heavy/refined_pixels"
    images_dir = "../data_heavy/frames_ear_only"
    for idx in lines:
        left_image = cv2.imread("%s/0-%s.png" % (images_dir, idx))
        right_image = cv2.imread("%s/1-%s.png" % (images_dir, idx))
        with open("%s/0-%s.png" % (refined_pixel_dir, idx), "rb") as fp:
            left_pixels = pickle.load(fp)
        with open("%s/1-%s.png" % (refined_pixel_dir, idx), "rb") as fp:
            right_pixels = pickle.load(fp)
        corr1 = match_zncc(left_image, right_image, left_pixels, right_pixels)
        corr2 = match_zncc(right_image, left_image, right_pixels, left_pixels)


if __name__ == '__main__':
    main()