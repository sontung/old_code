import os
import sys

import cv2
import numpy as np
from glob import glob


def sample_ellipse(rcenter, rw, rh, angle):
    rx = rw/2
    ry = rh/2
    radian_angle = np.radians(angle)
    alpha = np.linspace(0, 2*np.pi, 100,endpoint=True)
    points = []
    for al in alpha:
        x = rx*np.cos(al)*np.cos(radian_angle) - ry*np.sin(al)*np.sin(radian_angle) + rcenter[0]
        y = rx*np.cos(al)*np.sin(radian_angle) + ry*np.sin(al)*np.cos(radian_angle) + rcenter[1]
        points.append((int(x), int(y)))
    return points


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


def fit_ellipse(image, du):
    indices = np.nonzero(image[:,:,0])
    nz_points = np.transpose((indices[1], indices[0])).astype(int)
    contours = nz_points.reshape((nz_points.shape[0], 1, nz_points.shape[1]))
    ellipse = cv2.fitEllipse(contours)

    center = ellipse[0]
    w, h = ellipse[1]
    angle = ellipse[2]

    # sample_points = sample_ellipse(center, w, h, angle)
    # print(sample_points)
    # for u, v in sample_points:
    #     img[v, u] = (0, 0, 255)

    bias = 0.2
    for p in nz_points:
        dis = parametric_ellipse(center, w, h, angle, p)
        if dis <= bias:
            image[p[1], p[0]] = (0, 0, 0)
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

    # cv2.ellipse(image, ellipse, (0, 255, 0), 2)
    # cv2.imshow('cnts', image)
    # cv2.imwrite(f"/home/sontung/Desktop/img_test/img_{du}.png", image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    dum = 0
    for file in glob('../data_heavy/head_rotations/*.png'):
        # if dum == 111:
        #     print(file)
        img = cv2.imread(file)
        fit_ellipse(img, dum)
        dum += 1
        # sys.exit()