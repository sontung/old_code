from functools import partial
import matplotlib.pyplot as plt
from pycpd import DeformableRegistration, AffineRegistration
import numpy as np
import time


def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], color='blue', label='Source')
    plt.text(0.87, 0.92, 'Iteration: {:d}'.format(
        iteration), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)


def main():
    import cv2

    ear = cv2.imread("/home/sontung/work/3d-air-bag-p2/data/ear.png")
    # ear = cv2.resize(ear, (ear.shape[1]//4, ear.shape[0]//4))

    nonzero_indices = np.nonzero(ear)
    with open("../data/ear.txt", "w") as fp:
        for i in range(nonzero_indices[0].shape[0]):
            print(nonzero_indices[0][i]/ear.shape[0],
                  nonzero_indices[1][i]/ear.shape[1], file=fp)

    Y = np.loadtxt('../data/ear.txt')
    Y2 = np.loadtxt('../data/ear.txt')

    X = np.loadtxt('/home/sontung/work/3d-air-bag-p2/data_heavy/transformed/0-237.png.txt')

    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    callback = partial(visualize, ax=fig.axes[0])

    reg = AffineRegistration(**{'X': X, 'Y': Y}, tolerance=0.1)
    reg.register()
    Y = reg.transform_point_cloud(Y)

    ax = fig.axes[0]
    ax.scatter(X[:, 0],  X[:, 1], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], color='blue', label='Source')
    ax.scatter(Y2[:, 0],  Y2[:, 1], color='yellow', label='original')

    ax.legend(loc='upper left', fontsize='x-large')
    plt.show()


if __name__ == '__main__':
    main()