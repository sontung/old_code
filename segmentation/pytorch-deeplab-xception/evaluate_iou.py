import sys
import cv2
import numpy as np

from PIL import Image

import glob


def compute_iou(gt_image, pre_image, num_class=21):
    assert gt_image.shape == pre_image.shape
    _mask = (gt_image >= 0) & (gt_image < num_class)
    label = num_class * gt_image[_mask].astype('int') + pre_image[_mask]
    count = np.bincount(label, minlength=num_class ** 2)
    confusion_matrix = count.reshape(num_class, num_class)
    MIoU = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
            np.diag(confusion_matrix))
    MIoU = np.nanmean(MIoU)
    return MIoU


def evaluate(gt_dir="/home/sontung/work/to_Tung/test_ab_head",
             pred_dir="/home/sontung/work/to_Tung/predictions_ab_head"):
    gt_images = glob.glob(f"{gt_dir}/*.png")
    arr = []
    for gt_img_name in gt_images:
        pred_img_name = gt_img_name.split("/")[-1]
        gt_img = Image.open(gt_img_name)
        pred_img = Image.open(f"{pred_dir}/{pred_img_name}")
        gt_img = np.array(gt_img)
        pred_img = np.array(pred_img)
        score = compute_iou(gt_img, pred_img)
        arr.append(score)
    print(f"mIoU = {np.mean(arr)}")


def draw_image_by_mask(mask):
    print(np.unique(mask))
    res = np.zeros((mask.shape[0], mask.shape[1], 3))
    res[mask == 1, :] = [128, 255, 10]
    res[mask == 3, :] = [255, 0, 128]
    return res


def visualize_pred_test(gt_dir="/home/sontung/work/to_Tung/test_ab_head",
                        pred_dir="/home/sontung/work/to_Tung/predictions_ab_head"):
    gt_images = glob.glob(f"{gt_dir}/*.png")
    for gt_img_name in gt_images:
        pred_img_name = gt_img_name.split("/")[-1]
        gt_img = Image.open(gt_img_name)
        pred_img = Image.open(f"{pred_dir}/{pred_img_name}")
        gt_img = np.array(gt_img)
        pred_img = np.array(pred_img)
        res_im = np.hstack([draw_image_by_mask(pred_img), draw_image_by_mask(gt_img)])
        res_im = cv2.resize(res_im, (res_im.shape[1]//4, res_im.shape[0]//4))
        cv2.imshow("t", res_im)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize_pred_test()
    # evaluate()
