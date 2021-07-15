import sys
import pickle
import skimage.io
import torch
import numpy as np
import cv2
import medpy.filter
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.segmentation import chan_vese, morphological_geodesic_active_contour
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries, inverse_gaussian_gradient
from skimage.filters import sobel
from skimage.util import img_as_float
from skimage.io import imread
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

def keep_one_biggest_contour(img):
    """
    returns img (output where smaller contours removed), mask (where kept pixels are ones)
    """
    cnts, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = None
    if len(cnts) > 0:
        largest_cnt = max(cnts, key=lambda du1: cv2.contourArea(du1))
        res = np.zeros_like(img)
        mask = np.zeros([img.shape[0], img.shape[1], 3])
        cv2.fillPoly(res, pts=[largest_cnt], color=(192, 128, 128))
        img[res != (192, 128, 128)] = 0
        mask[res == (192, 128, 128)] = 1
        mask = mask[:, :, 0]
    return img, mask


def post_process(prediction, rgb_image, name):
    # airbag
    ab_pixels = np.transpose(np.nonzero(prediction == 15))
    img_ab = np.zeros_like(prediction, np.uint8)
    img_ab[ab_pixels[:, 0], ab_pixels[:, 1]] = 1
    cnts, _ = cv2.findContours(img_ab, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(prediction.shape, rgb_image.shape)
    rgb_image = rgb_image.astype(np.uint8)
    if len(cnts) > 0:
        largest_cnt = max(cnts, key=lambda du1: cv2.contourArea(du1))
        # refine_mask(largest_cnt, rgb_image)
        # cv2.drawContours(rgb_image, [largest_cnt], -1, (0, 255, 0), 3)
        # cv2.imshow("t", rgb_image)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        cv2.imwrite(f"snakes/{name}.png", rgb_image)
        largest_cnt = largest_cnt.reshape((-1, 2))
        np.savetxt(f"snakes/{name}.npy", largest_cnt)


def snake_contour():
    for i in range(1, 801):
        try:
            org_img = skimage.io.imread(f"snakes/{i}.png")
            contour = np.loadtxt(f"snakes/{i}.npy").astype(np.int64)
        except:
            continue

        init = np.zeros_like(org_img)
        cv2.drawContours(init, [contour], -1, (1, 1, 1), 3)

        rect = cv2.minAreaRect(contour)

        box = cv2.boxPoints(rect)
        box = np.int0(box)
        bbox_init = np.zeros(org_img.shape, dtype=np.uint8)
        cv2.drawContours(bbox_init, [box], -1, (1, 1, 1), 3)

        box = box.reshape((box.shape[0], -1, box.shape[1]))
        color = [192, 128, 128]
        mask = np.zeros(org_img.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [box], color)
        nonzero_index = np.transpose(np.nonzero(mask[:, :, 0]))
        bbox_only_rgb_img = np.zeros(org_img.shape, dtype=np.uint8)
        bbox_only_rgb_img[nonzero_index[:, 0], nonzero_index[:, 1], :] = org_img[nonzero_index[:, 0], nonzero_index[:, 1], :]

        out1 = kmeans_segment_on_grad_pixels(bbox_only_rgb_img, nonzero_index)
        out2 = kmeans_segment_on_grad(org_img)

        cv2.imshow("t", np.hstack([bbox_only_rgb_img, org_img, out1, out2]))
        cv2.waitKey()
        cv2.destroyAllWindows()

        test_mask = np.zeros(org_img.shape, dtype=np.uint8)
        cv2.fillPoly(test_mask, [contour], color)
        nonzero_index2 = np.transpose(np.nonzero(test_mask[:, :, 0]))
        mask_pixels = org_img[nonzero_index2[:, 0], nonzero_index2[:, 1], :]
        mask_pixels_added = np.zeros((mask_pixels.shape[0] + 1, 3))
        mask_pixels_added[:mask_pixels.shape[0]] = mask_pixels

        mask[nonzero_index[:, 0], nonzero_index[:, 1], :] = [128, 255, 255]
        mask[nonzero_index2[:, 0], nonzero_index2[:, 1], :] = [128, 128, 128]
        candidate_pixels = np.transpose(np.nonzero((mask == [128, 255, 255]).all(axis=2)))
        mask[nonzero_index[:, 0], nonzero_index[:, 1], :] = [0, 0, 0]
        mask[nonzero_index2[:, 0], nonzero_index2[:, 1], :] = [128, 128, 128]


def kmeans_segment_on_grad(img):
    img = img_as_float(img)
    gimg = inverse_gaussian_gradient(img)
    out = k_means_smoothing(gimg)
    return out

def kmeans_segment_on_grad_pixels(img, pixels):
    img = img_as_float(img)
    gimg = inverse_gaussian_gradient(img)
    img_arr = [gimg[u, v] for u, v in pixels]
    image_array_sample = shuffle(img_arr, random_state=0)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(image_array_sample)
    labels = kmeans.predict(img_arr)
    codebook = {0: [128, 128, 0], 1: [255, 1, 0]}
    w, h, d = tuple(img.shape)
    image = np.zeros((w, h, d))
    for i, (u, v) in enumerate(pixels):
        image[u][v] = codebook[labels[i]]

    return image.astype(np.uint8)

def k_means_smoothing(rgb):
    n_colors = 2

    china = rgb
    china = np.array(china, dtype=np.float64) / 255
    w, h, d = tuple(china.shape)
    assert d == 3
    image_array = np.reshape(china, (w * h, d))

    image_array_sample = shuffle(image_array, random_state=0)
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

    # Get labels for all points
    labels = kmeans.predict(image_array)
    codebook = {0: [128, 128, 0], 1: [255, 1, 0]}
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1

    return image.astype(np.uint8)


def segmentation_algo(img):
    simg = medpy.filter.smoothing.anisotropic_diffusion(img, option=3, niter=20)
    simg = simg.astype(np.uint8)

    # cv2.imshow("t", np.hstack([img, simg]))
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    img = img_as_float(simg)
    segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=1000)
    return segments_fz
    segments_slic = slic(img, n_segments=50, compactness=10, sigma=1,
                         start_label=1)
    segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
    gradient = sobel(rgb2gray(img))
    segments_watershed = watershed(gradient, markers=250, compactness=0.001)

    print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}")
    print(f"SLIC number of segments: {len(np.unique(segments_slic))}")
    print(f"Quickshift number of segments: {len(np.unique(segments_quick))}")

    fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    ax[0, 0].imshow(mark_boundaries(img, segments_fz))
    ax[0, 0].set_title("Felzenszwalbs's method")
    ax[0, 1].imshow(mark_boundaries(img, segments_slic))
    ax[0, 1].set_title('SLIC')
    ax[1, 0].imshow(mark_boundaries(img, segments_quick))
    ax[1, 0].set_title('Quickshift')
    ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
    ax[1, 1].set_title('Compact watershed')
    for a in ax.ravel():
        a.set_axis_off()

    plt.tight_layout()
    plt.show()
    return [mark_boundaries(img, segments_fz)]


def chan_vess(img):
    import matplotlib.pyplot as plt
    from skimage import data, img_as_float
    from skimage.segmentation import chan_vese
    import skimage.color
    img = skimage.color.rgb2gray(img)
    image = img_as_float(img)
    # Feel free to play around with the parameters to see how they impact the result
    cv = chan_vese(image, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_iter=200,
                   dt=0.5, init_level_set="checkerboard", extended_output=True)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    ax = axes.flatten()

    ax[0].imshow(image, cmap="gray")
    ax[0].set_axis_off()
    ax[0].set_title("Original Image", fontsize=12)

    ax[1].imshow(cv[0], cmap="gray")
    ax[1].set_axis_off()
    title = "Chan-Vese segmentation - {} iterations".format(len(cv[2]))
    ax[1].set_title(title, fontsize=12)

    ax[2].imshow(cv[1], cmap="gray")
    ax[2].set_axis_off()
    ax[2].set_title("Final Level Set", fontsize=12)

    ax[3].plot(cv[2])
    ax[3].set_title("Evolution of energy over iterations", fontsize=12)

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    snake_contour()
