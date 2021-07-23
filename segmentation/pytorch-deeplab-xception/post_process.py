import numpy as np
import cv2


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

