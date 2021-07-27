import numpy as np


def laplacian_fairing(angles):
    angles = check(angles)
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


def check(angles):
    print("final check before laplacian fairing")
    if angles[0] is None:
        for i in range(1, len(angles)):
            if angles[i] is not None:
                angles[0] = angles[i]
                print(f" sub 0-th index by {i}-th index of value {angles[i]}")
                break
    if angles[1] is None:
        for i in range(2, len(angles)):
            if angles[i] is not None:
                angles[1] = angles[i]
                print(f" sub 1-th index by {i}-th index of value {angles[i]}")
                break
    if angles[len(angles)-1] is None:
        for i in range(len(angles)-2, 0, -1):
            if angles[i] is not None:
                angles[len(angles)-1] = angles[i]
                print(f" sub {len(angles)-1}-th index by {i}-th index of value {angles[i]}")
                break
    if angles[len(angles)-2] is None:
        for i in range(len(angles)-3, 0, -1):
            if angles[i] is not None:
                angles[len(angles)-2] = angles[i]
                print(f" sub {len(angles)-2}-th index by {i}-th index of value {angles[i]}")
                break
    return angles
