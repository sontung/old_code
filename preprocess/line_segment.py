import os
import sys

import cv2
import numpy as np
from glob import glob


def sample_ellipse(rcenter, rw, rh, angle):
    rx = rw/2
    ry = rh/2
    radian_angle = np.radians(angle)
    alpha = np.linspace(0, 2*np.pi, 100, endpoint=True)
    points = []
    for al in alpha:
        x = rx*np.cos(al)*np.cos(radian_angle) - ry*np.sin(al)*np.sin(radian_angle) + rcenter[0]
        y = rx*np.cos(al)*np.sin(radian_angle) + ry*np.sin(al)*np.cos(radian_angle) + rcenter[1]
        points.append((int(x), int(y)))
    return points
