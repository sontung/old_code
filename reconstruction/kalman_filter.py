import numpy as np
import time
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from rec_utils import b_spline_smooth
from matplotlib import pyplot as plt


def normalize_angle(x):
    x = np.radians(x)
    x = x % (2 * np.pi)
    if x > np.pi:
        x -= 2 * np.pi
    return np.degrees(x)


def fx(x, dt, u):
    new_states = x[:]
    if u[0] > 0:
        new_states[2] = u[0] / 2
    else:
        new_states[2] = u[1]
    new_states[1] = x[1] + x[2]*dt
    new_states[0] = x[0] + x[1]*dt
    return new_states


def hx(state_):
    return np.array([state_[0]])


def residual_h(a, b):
    y = a - b
    y[0] = normalize_angle(y[0])
    return y


def residual_x(a, b):
    y = a - b
    y[0] = normalize_angle(y[0])
    y[1] = normalize_angle(y[1])
    y[2] = normalize_angle(y[2])
    return y


def kalman_smooth(angles, angles_spline):

    dt = 1
    points = MerweScaledSigmaPoints(3, alpha=.1, beta=2., kappa=-1)

    ukf = UnscentedKalmanFilter(
        dim_x=3, dim_z=1, fx=fx, hx=hx, dt=dt, points=points,
        residual_x=residual_x, residual_z=residual_h)
    ukf.x = np.array([0, 0, 0])
    ukf.P = np.diag([.1, .1, .05])
    ukf.Q = np.eye(3)*0.0001

    # angles = [-3.4462777027866776, -2.1150250755250406, -1.8560732528568555, -1.3574491120418282, -1.9546781171926215,
    #           -3.2734472716413046, -1.0872772653182894, -3.5105807534448528, -1.630900699090774, -0.7040869824998134,
    #           -2.4389707939738656, -1.7013383263965176, -1.700699473614145, -3.2926932466343324, -2.682874266060129,
    #           -1.976978904131933, -3.255259406658046, -3.1089576073968206, -3.0907377835695913, -3.9150016407316484,
    #           -2.608541079583464, -3.1756303516994775, -3.576447407041296, -3.5430354249761327, -1.7217838073417828,
    #           -2.7387528621254464, -2.982519293322194, -1.7270068806145518, -0.14698977255068216, 3.152358527307376,
    #           6.110672974270201, 11.282876184188087, 16.731277354001755, 19.247687601717978, None, None, None, None, None,
    #           None, None, None, None, None, None, None, None, None, None, None, 22.688711753964206, 6.163171894580451,
    #           5.287295527264798, 14.621030445863529, 24.241563554687293, 26.978299546281132, 27.72163226221281,
    #           -39.679278262916284, 26.182223906695505, 12.20184900439492, 17.931186169334115, 13.889772675802327,
    #           8.420033492333705, None, 2.432931184129019, -1.7305925821014478, 0.35896367442838506, -3.6254288311620755,
    #           -11.125160058885974, -12.628546710957924, -15.747831931260873, -19.597861043749145, -22.81247066292442,
    #           -25.58436698968336, -28.538474775553823, -25.3960300152347, -16.456842931264294, 6.685207752747491,
    #           19.488347538793196, 27.551517685662347, 26.075859374935472, 27.062517178012236, 6.011572279737814,
    #           26.892404943686618, -63.28764972386039, 20.558372192743466, 22.993873603607625, 7.7387799841224005]

    kp = 0.55
    ki = 0.01
    kd = 0.5
    e = 0
    e_sum = 0
    e_prev = 0
    crash = False
    kalman_angles = []
    for angle in angles:
        u = [0, 0]
        if angle is None and not crash:
            u[0] = 50
            crash = True
        elif crash:
            e_sum += e * dt
            dedt = (e - e_prev) / dt
            u[1] = kp * e + ki * e_sum + kd * dedt
            e_prev = e

        ukf.predict(u=u)
        ukf.update(angle)
        kalman_angles.append(ukf.x[0])
        e = normalize_angle(-ukf.x[0])

    plt.plot(kalman_angles, "r")
    plt.plot(angles_spline, "b")
    plt.legend(["kalman", "b-spline"])
    plt.savefig(f"kalman-{time.time()}.png")