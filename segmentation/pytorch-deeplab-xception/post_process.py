import torch
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

def post_process(prediction, rgb_image):
    # airbag
    print(prediction.shape, rgb_image.shape)
    ab_pixels = np.transpose(np.nonzero(prediction == 15))
    img_ab = np.zeros_like(prediction, np.uint8)
    img_ab[ab_pixels[:, 0], ab_pixels[:, 1]] = 1
    #img_ab, _ = keep_one_biggest_contour(img_ab)
    cnts, _ = cv2.findContours(img_ab, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        largest_cnt = max(cnts, key=lambda du1: cv2.contourArea(du1))
        refine_mask(largest_cnt, rgb_image)

def refine_mask(contour, org_img, vis=True):
    org_img_copy = org_img.copy()

    mask = np.zeros(org_img.shape, dtype=np.uint8)
    rect = cv2.minAreaRect(contour)

    box = cv2.boxPoints(rect)
    box = np.int0(box)
    box = box.reshape((box.shape[0], -1, box.shape[1]))
    color = [192, 128, 128]
    cv2.fillPoly(mask, [box], color)
    nonzero_index = np.transpose(np.nonzero(mask[:, :, 0]))

    test_mask = np.zeros(org_img.shape, dtype=np.uint8)
    cv2.fillPoly(test_mask, [contour], color)
    nonzero_index2 = np.transpose(np.nonzero(test_mask[:, :, 0]))
    mask_pixels = org_img_copy[nonzero_index2[:, 0], nonzero_index2[:, 1], :]
    mask_pixels_added = np.zeros((mask_pixels.shape[0]+1, 3))
    mask_pixels_added[:mask_pixels.shape[0]] = mask_pixels

    mask[nonzero_index[:, 0], nonzero_index[:, 1], :] = [128, 255, 255]
    mask[nonzero_index2[:, 0], nonzero_index2[:, 1], :] = [128, 128, 128]
    candidate_pixels = np.transpose(np.nonzero((mask == [128, 255, 255]).all(axis=2)))
    mask[nonzero_index[:, 0], nonzero_index[:, 1], :] = [0, 0, 0]
    mask[nonzero_index2[:, 0], nonzero_index2[:, 1], :] = [128, 128, 128]

    candidate_pixels_gpu = torch.from_numpy(candidate_pixels).cuda()
    mask_pixels_added_gpu = torch.from_numpy(mask_pixels_added).cuda()
    mask_pixels_gpu = torch.from_numpy(mask_pixels).float().cuda()
    best_var = torch.var(mask_pixels_gpu)
    org_img_copy_gpu = torch.from_numpy(org_img_copy).cuda()
    test_mask = torch.from_numpy(test_mask).cuda()
    for i, p in enumerate(candidate_pixels_gpu):
        mask_pixels_added_gpu[-1, :] = org_img_copy_gpu[p[0], p[1]]
        if torch.var(mask_pixels_added_gpu) <= best_var:
            test_mask[p[0], p[1]] = 255
    test_mask = test_mask.cpu().numpy()

    if vis:
        org_img = org_img.astype(np.uint8)
        blend2 = cv2.addWeighted(org_img, 0.7, test_mask, 0.3, 0)
        res = np.hstack([test_mask, blend2, org_img])
        res = cv2.resize(res, [res.shape[1]//4, res.shape[0]//4])

        cv2.imshow("t", res)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return mask
