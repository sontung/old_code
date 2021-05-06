import sys
import pickle
import cv2
from tqdm import tqdm
import imageio
import numpy as np
from utils import compute_zncc
from pathlib import Path


def match_zncc(im1, im2, pixels1, pixels2, name):
    """
    zncc based matching
    """
    corr = {}

    global_mean1 = [np.mean(im1[:, :, c]) for c in range(3)]
    global_mean2 = [np.mean(im2[:, :, c]) for c in range(3)]
    saved_solution_dir = "../data_heavy/matching_solutions"
    a_file = Path("%s/%s" % (saved_solution_dir, name))
    if a_file.exists():
        with open("%s/%s" % (saved_solution_dir, name), "rb") as fp:
            corr = pickle.load(fp)
            return corr
    else:
        for x, y in tqdm(pixels1, desc="[%s] Matching %d pixels with %d pixels" % (name, len(pixels1), len(pixels2))):
            a_match = None
            best_score = 0
            for x2, y2 in pixels2:
                score, w1, w2 = compute_zncc(x, y, x2, y2, im1, im2, global_mean1, global_mean2, 25)
                if score > best_score:
                    best_score = score
                    a_match = (x2, y2, w1, w2)
            x2, y2, w1, w2 = a_match
            corr[(x, y)] = (x2, y2)

            # print(x, y, x2, y2, best_score)
            # im1_d = im1.copy()
            # im2_d = im2.copy()
            # cv2.circle(im1_d, (y, x), 20, (125, 125, 125), 2)
            # cv2.circle(im2_d, (y2, x2), 20, (125, 125, 125), 2)
            # im1_d = cv2.resize(im1_d, (im1_d.shape[1]//4, im1_d.shape[0]//4))
            # im2_d = cv2.resize(im2_d, (im2_d.shape[1]//4, im2_d.shape[0]//4))
            # cv2.imshow("test", np.hstack([im1_d, im2_d]))
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            # sys.exit()

        with open("%s/%s" % (saved_solution_dir, name), "wb") as fp:
            pickle.dump(corr, fp)
        return corr


def consistency_check(match1, match2):
    """
    consistency check when matching in both directions
    """
    res = {}
    for p in match1:
        q = match1[p]
        p2 = match2[q]
        if p == p2:
            res[p] = q
    return res


def main():
    sys.stdin = open("../data_heavy/frames/info.txt")
    lines = [du[:-1] for du in sys.stdin.readlines()]
    refined_pixel_dir = "../data_heavy/refined_pixels"
    images_dir = "../data_heavy/frames_ear_only_nonblack_bg"
    for idx in lines:
        left_image = imageio.imread("%s/0-%s.png" % (images_dir, idx))
        right_image = imageio.imread("%s/1-%s.png" % (images_dir, idx))
        with open("%s/0-%s.png" % (refined_pixel_dir, idx), "rb") as fp:
            left_pixels = pickle.load(fp)
        with open("%s/1-%s.png" % (refined_pixel_dir, idx), "rb") as fp:
            right_pixels = pickle.load(fp)
        corr1 = match_zncc(left_image, right_image, left_pixels, right_pixels, "0-1-%s" % idx)
        corr2 = match_zncc(right_image, left_image, right_pixels, left_pixels, "1-0-%s" % idx)
        final_corr = consistency_check(corr1, corr2)
        print("final corr: %d matches" % (len(final_corr)))

        im1_d = left_image.copy()
        im2_d = right_image.copy()
        im3 = np.hstack([im1_d, im2_d])
        for x, y in final_corr:
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x2, y2 = final_corr[(x, y)]
            cv2.circle(im3, (y, x), 5, color, -1)
            cv2.circle(im3, (y2+im1_d.shape[1], x2), 5, color, -1)
            cv2.line(im3, (y, x), (y2+im1_d.shape[1], x2), color)
        cv2.imwrite("../data_heavy/matching_debugs/%s.png" % idx, im3)


if __name__ == '__main__':
    main()