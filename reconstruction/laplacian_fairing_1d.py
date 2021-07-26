import numpy as np
from matplotlib import pyplot as plt
import time


def laplacian_fairing(angles, angles_spline):
    mat_a = np.zeros((len(angles), len(angles))).astype(np.float64)
    mat_b = np.zeros((len(angles), 1)).astype(np.float64)
    measurements = [du for du in angles if du is not None]
    indices = [du for du in range(len(angles)) if angles[du] is not None]

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
    plt.plot(mat_x, "r")
    plt.plot(angles_spline, "b")
    plt.plot(indices, measurements, "ro")
    plt.legend(["laplacian", "b-spline"])
    plt.savefig(f"laplacian-{1}.png")
    plt.close()
    print(f"laplacian cost={cost}")
    return mat_x


if __name__ == '__main__':
    angles = [-3.4462777027866776, -2.1150250755250406, -1.8560732528568555, -1.3574491120418282, -1.9546781171926215,
              -3.2734472716413046, -1.0872772653182894, -3.5105807534448528, -1.630900699090774, -0.7040869824998134,
              -2.4389707939738656, -1.7013383263965176, -1.700699473614145, -3.2926932466343324, -2.682874266060129,
              -1.976978904131933, -3.255259406658046, -3.1089576073968206, -3.0907377835695913, -3.9150016407316484,
              -2.608541079583464, -3.1756303516994775, -3.576447407041296, -3.5430354249761327, -1.7217838073417828,
              -2.7387528621254464, -2.982519293322194, -1.7270068806145518, -0.14698977255068216, 3.152358527307376,
              6.110672974270201, 11.282876184188087, 16.731277354001755, 19.247687601717978, None, None, None, None, None,
              None, None, None, None, None, None, None, None, None, None, None, 22.688711753964206, 6.163171894580451,
              5.287295527264798, 14.621030445863529, 24.241563554687293, 26.978299546281132, 27.72163226221281,
              -39.679278262916284, 26.182223906695505, 12.20184900439492, 17.931186169334115, 13.889772675802327,
              8.420033492333705, None, 2.432931184129019, -1.7305925821014478, 0.35896367442838506, -3.6254288311620755,
              -11.125160058885974, -12.628546710957924, -15.747831931260873, -19.597861043749145, -22.81247066292442,
              -25.58436698968336, -28.538474775553823, -25.3960300152347, -16.456842931264294, 6.685207752747491,
              19.488347538793196, 27.551517685662347, 26.075859374935472, 27.062517178012236, 6.011572279737814,
              26.892404943686618, -63.28764972386039, 20.558372192743466, 22.993873603607625, 7.7387799841224005]

    laplacian_fairing(angles, [])