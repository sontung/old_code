import numpy as np


def laplacian_fairing(angles):
    mat_a = np.zeros((len(angles), len(angles))).astype(np.float64)
    mat_b = np.zeros((len(angles), 1)).astype(np.float64)

    for angle_idx, angle in enumerate(angles):
        if angle is not None:
            mat_a[angle_idx][angle_idx] = 1
            mat_b[angle_idx] = angle
        else:
            mat_b[angle_idx] = 0
            mat_a[angle_idx][angle_idx-2] = 1
            mat_a[angle_idx][angle_idx-1] = -4
            mat_a[angle_idx][angle_idx] = 6
            mat_a[angle_idx][angle_idx+1] = -4
            mat_a[angle_idx][angle_idx+2] = 1

    mat_x = np.linalg.solve(mat_a, mat_b).reshape((-1,))
    cost = 0
    for angle_idx, angle in enumerate(angles):
        if angle is not None:
            continue
        else:
            cost += mat_x[angle_idx-2]-4*mat_x[angle_idx-1]+6*mat_x[angle_idx]-4*mat_x[angle_idx+1]+mat_x[angle_idx+2]
    print(f"laplacian cost={cost}")
    return mat_x
