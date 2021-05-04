import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

from ellipse import LsqEllipse
from matplotlib.patches import Ellipse


def test1():
    dtype = torch.float
    device = torch.device("cuda:0")  # Uncomment this to run on GPU
    # device = torch.device("cuda:0")  # Uncomment this to run on GPU

    rx = 5
    ry = 7
    cx = 3
    cy = 9
    theta = 0.523599

    t = torch.linspace(0, math.pi, 2000, device=device, dtype=dtype)
    x = rx * torch.cos(t) * np.cos(theta) - ry * torch.sin(t) * np.sin(theta) + cx
    y = rx * torch.cos(t) * np.sin(theta) + ry * torch.sin(t) * np.cos(theta) + cy
    plt.subplot(411)
    plt.plot(x.cpu(), y.cpu(), "g")

    t = torch.linspace(0, 2*math.pi, 2000, device=device, dtype=dtype)
    x = rx * torch.cos(t) * np.cos(theta) - ry * torch.sin(t) * np.sin(theta) + cx
    y = rx * torch.cos(t) * np.sin(theta) + ry * torch.sin(t) * np.cos(theta) + cy
    plt.subplot(412)
    plt.plot(x.cpu(), y.cpu(), "y")

    a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

    x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
    y = torch.sin(x)
    plt.subplot(413)
    plt.plot(x.cpu(), y.cpu(), "b")
    x = torch.linspace(0, 2*math.pi, 2000, device=device, dtype=dtype)
    y = torch.sin(x)

    plt.subplot(414)
    plt.plot(x.cpu(), y.cpu(), "r")
    plt.show()

    learning_rate = 1e-6
    for t in range(10):
        # Forward pass: compute predicted y using operations on Tensors.
        y_pred = a + b * x + c * x ** 2 + d * x ** 3

        # Compute and print loss using operations on Tensors.
        # Now loss is a Tensor of shape (1,)
        # loss.item() gets the scalar value held in the loss.
        loss = (y_pred - y).pow(2).sum()
        print("loss", loss.item())

        # Use autograd to compute the backward pass. This call will compute the
        # gradient of loss with respect to all Tensors with requires_grad=True.
        # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
        # the gradient of the loss with respect to a, b, c, d respectively.
        loss.backward()

        # Manually update weights using gradient descent. Wrap in torch.no_grad()
        # because weights have requires_grad=True, but we don't need to track this
        # in autograd.
        with torch.no_grad():
            a -= learning_rate * a.grad
            b -= learning_rate * b.grad
            c -= learning_rate * c.grad
            d -= learning_rate * d.grad

            # Manually zero the gradients after updating weights
            a.grad = None
            b.grad = None
            c.grad = None
            d.grad = None

    print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')


def make_test_ellipse(center=[1, 1], width=1, height=.6, phi=3.14/5):
    """Generate Elliptical data with noise
    Parameters
    ----------
    center: list:float
        (<x_location>, <y_location>)
    width: float
        semimajor axis. Horizontal dimension of the ellipse (**)
    height: float
        semiminor axis. Vertical dimension of the ellipse (**)
    phi: float:radians
        tilt of the ellipse, the angle the semimajor axis
        makes with the x-axis
    Returns
    -------
    data:  list:list:float
        list of two lists containing the x and y data of the ellipse.
        of the form [[x1, x2, ..., xi],[y1, y2, ..., yi]]
    """
    t = np.linspace(0, 2*np.pi, 1000)
    x_noise, y_noise = np.random.rand(2, len(t))

    ellipse_x = center[0] + width*np.cos(t)*np.cos(phi)-height*np.sin(t)*np.sin(phi) + x_noise/2.  # noqa: E501
    ellipse_y = center[1] + width*np.cos(t)*np.sin(phi)+height*np.sin(t)*np.cos(phi) + y_noise/2.  # noqa: E501

    return [ellipse_x, ellipse_y]


def test3():

    X1, X2 = make_test_ellipse()

    rx = 5
    ry = 7
    cx = 3
    cy = 9
    theta = 0.523599

    t = torch.linspace(-math.pi, math.pi, 100, device="cpu", dtype=torch.float)
    x = rx * torch.cos(t) * np.cos(theta) - ry * torch.sin(t) * np.sin(theta) + cx
    y = rx * torch.cos(t) * np.sin(theta) + ry * torch.sin(t) * np.cos(theta) + cy
    X1 = x.numpy()
    X2 = y.numpy()

    X = np.array(list(zip(X1, X2)))
    reg = LsqEllipse().fit(X)
    center, width, height, phi = reg.as_parameters()

    print(f'center: {center[0]:.3f}, {center[1]:.3f}')
    print(f'width: {width:.3f}')
    print(f'height: {height:.3f}')
    print(f'phi: {phi:.3f}')

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot()
    ax.axis('equal')
    ax.plot(X1, X2, 'ro', zorder=1)
    ellipse = Ellipse(
        xy=center, width=2 * width, height=2 * height, angle=np.rad2deg(phi),
        edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
    )
    ax.add_patch(ellipse)

    plt.xlabel('$X_1$')
    plt.ylabel('$X_2$')

    plt.legend()
    plt.show()


def loss_func(x, y, ellipse):
    (cx, cy), rx, ry, theta = ellipse
    p1 = np.power((x - cx) * np.cos(theta) + (y - cy) * np.sin(theta), 2) / rx**2
    p2 = np.power((x - cx) * np.sin(theta) - (y - cy) * np.cos(theta), 2) / ry**2
    loss = np.sum(np.abs(p1 + p2))
    return loss, np.sum((p1 + p2-1)**2)


def fitting_algo(data, threshold=0.1):
    reg = LsqEllipse().fit(data)
    _, l2_total = loss_func(data[:, 0], data[:, 1], reg.as_parameters())
    if l2_total <= threshold:
        return reg.as_parameters()
    while True:
        data2 = []
        for i in range(len(data)):
            l1, l2 = loss_func(data[i, 0], data[i, 1], reg.as_parameters())
            if l1 > 1:
                data2.append(data[i])
        data2 = np.array(data2)
        reg = LsqEllipse().fit(data2)
        _, l2_total = loss_func(data2[:, 0], data2[:, 1], reg.as_parameters())
        if l2_total <= threshold:
            return reg.as_parameters()


def test4():
    img = cv2.imread("edges.png")
    # X1 = []
    # X2 = []
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         if img[i, j, 0] == 255:
    #             X1.append(i)
    #             X2.append(j)
    #
    # X = np.array(list(zip(X1, X2)))
    # np.save('data.npy', X)

    X = np.load('data.npy')

    center, width, height, phi = fitting_algo(X)
    center = tuple(map(int, center))
    width, height = map(int, [width, height])
    phi = np.rad2deg(phi)

    cv2.ellipse(img, (center[1], center[0]), (height, width), phi, 0, 360, 255, 10)
    cv2.imshow("test", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test4()
