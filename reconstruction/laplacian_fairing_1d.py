import numpy as np
from matplotlib import pyplot as plt


def laplacian_fairing(angles, name=None, collision_interval=None):
    """
    solve for missing values by minimizing the second derivative
    angles: np.array
    name: string
    collision_interval: tuple
    mat_x: np.array
    """

    # fill in values in collision period
    if collision_interval is not None:
        start, end = collision_interval
        path = angles[start: end]
        if angles[start - 1] is not None:
            if end == len(angles):
                path = [angles[start-1] for _ in range(len(path))]
            elif end < len(angles) and angles[end] is not None:
                path = np.linspace(angles[start-1], angles[end], len(path))
            angles[start: end] = path

    angles = check(angles)  # if two ends of the list are nones, modify
    mat_a = np.zeros((len(angles), len(angles))).astype(np.float64)
    mat_b = np.zeros((len(angles), 1)).astype(np.float64)

    # create matrix A
    for angle_idx, angle in enumerate(angles):
        if angle is not None:
            mat_a[angle_idx][angle_idx] = 1
            mat_b[angle_idx] = angle
        else:
            # neighboring coefficients for fxx = 0
            mat_b[angle_idx] = 0
            mat_a[angle_idx][angle_idx-2] = 1
            mat_a[angle_idx][angle_idx-1] = -4
            mat_a[angle_idx][angle_idx] = 6
            mat_a[angle_idx][angle_idx+1] = -4
            mat_a[angle_idx][angle_idx+2] = 1

    mat_x = np.linalg.solve(mat_a, mat_b).reshape((-1,))
    if name is not None:
        plt.plot(mat_x)
        plt.plot(angles, "bo")
        plt.savefig(name)
        plt.close()
    return mat_x


def check(angles):
    """
    check if the input has Nones in two places of its both ends
    """
    if angles[0] is None:
        for i in range(1, len(angles)):
            if angles[i] is not None:
                angles[0] = angles[i]
                break
    if angles[1] is None:
        for i in range(2, len(angles)):
            if angles[i] is not None:
                angles[1] = angles[i]
                break
    if angles[len(angles)-1] is None:
        for i in range(len(angles)-2, 0, -1):
            if angles[i] is not None:
                angles[len(angles)-1] = angles[i]
                break
    if angles[len(angles)-2] is None:
        for i in range(len(angles)-3, 0, -1):
            if angles[i] is not None:
                angles[len(angles)-2] = angles[i]
                break
    return angles
