import glob
import os
import pickle
from argparse import ArgumentParser
import sys
import cv2
import numpy as np
import kmeans1d
from PIL import Image
from tqdm import tqdm

from mmseg.apis import inference_segmentor, init_segmentor

parser = ArgumentParser()
parser.add_argument('-d', '--debug', type=bool, default=False, help='Debug mode')
parser.add_argument('-v', '--visualize', type=bool, default=False, help="Visualization mode")
args = vars(parser.parse_args())
DEBUG_MODE = args['debug']
VISUALIZE_MODE = args["visualize"]

# section Runtime Arguments


BACKGROUND, AIRBAG, SIDE_AIRBAG, HEAD, EAR = 0, 1, 2, 3, 4
HEAD_COLOR = [64, 128, 128]
AIRBAG_COLOR = [192, 128, 128]
EAR_COLOR = [255, 153, 51]
DEBUG = False


def merge_2images(_im1, _im2, c1, c2):
    output = np.zeros(_im1.shape, dtype=np.uint8)

    indices = np.argwhere(_im1[:, :] == c1)
    output[indices[:, 0], indices[:, 1]] = c1

    indices = np.argwhere(_im2[:, :] == c2)
    output[indices[:, 0], indices[:, 1]] = c2

    return output


def compute_position_and_rotation(mask, rgb_mask, contours):
    assert mask.shape == rgb_mask.shape[:2]
    x1, y1, x2, y2 = 0, 0, 0, 0

    if len(contours) < 1:
        return 0, None, None, x1, y1, x2, y2

    if len(contours) > 1:
        points = contours[0]
        for i in range(1, len(contours)):
            points = np.append(points, contours[i], axis=0)
        hull = cv2.convexHull(points)
        copy_mask = np.zeros(rgb_mask.shape, dtype=np.uint8)
        cv2.fillPoly(copy_mask, [hull], (255, 255, 255))
        copy_mask = copy_mask[:, :, 0]
        mask = copy_mask

    else:
        hull = contours[0]

    area = np.sum(mask == 255)

    orient_rect = cv2.minAreaRect(hull)
    center = (int(orient_rect[0][0]), int(orient_rect[0][1]))

    bb_box = cv2.boxPoints(orient_rect)
    bb_box = np.int0(bb_box)

    ind_by_x = np.argsort(bb_box[:, 1])
    top_points = bb_box[ind_by_x[:2]]
    bot_points = bb_box[ind_by_x[2:]]
    left_top_point = top_points[np.argmax(top_points[:, 0])]
    left_bot_point = bot_points[np.argmax(bot_points[:, 0])]

    x1, y1 = left_top_point
    x2, y2 = left_bot_point

    angle = np.rad2deg(np.arctan2(y2 - y1, x1 - x2))
    if angle < 0:
        angle += 180

    return area, center, angle, x1, y1, x2, y2


def refine_airbag_segmentation(ab_mask, head_mask):
    test_mask = cv2.bitwise_or(ab_mask, head_mask)
    test_cnts, _ = cv2.findContours(test_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    overlap_mask = None
    for cnt in test_cnts:
        dum_mask = np.zeros(head_mask.shape, dtype=np.uint8)
        cv2.fillPoly(dum_mask, [cnt], [255])
        intersection = cv2.bitwise_and(head_mask, dum_mask)
        intersect_pixel = np.sum(intersection)
        if intersect_pixel > 0:
            overlap_mask = dum_mask
            break
    if overlap_mask is None:
        return None

    refine_ab_mask = cv2.bitwise_and(ab_mask, overlap_mask)
    ab_contours, _ = cv2.findContours(refine_ab_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return refine_ab_mask, ab_contours


def get_seg_head_from_prediction(class_predict):
    head_mask = np.zeros(class_predict.shape, dtype=np.uint8)
    head_mask[class_predict == HEAD] = 255
    head_mask[class_predict == EAR] = 255
    rgb_mask = np.zeros((class_predict.shape[0], class_predict.shape[1], 3), dtype=np.uint8)

    contours, _ = cv2.findContours(head_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        max_contours = [max(contours, key=lambda x: cv2.contourArea(x))]
        cv2.fillPoly(rgb_mask, max_contours, HEAD_COLOR)
    else:
        max_contours = contours

    mask = rgb_mask[:, :, 0]
    mask[mask > 0] = 255

    return mask, rgb_mask, max_contours


def get_seg_airbag_from_prediction(class_predict, view=None, kept_area=1000):
    ab_mask = np.zeros(class_predict.shape, dtype=np.uint8)
    ab_mask[class_predict == AIRBAG] = 255
    rgb_mask = np.zeros((class_predict.shape[0], class_predict.shape[1], 3), dtype=np.uint8)

    contours, _ = cv2.findContours(ab_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        if view == 2:
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= kept_area]
        else:
            largest_cnt = max(contours, key=lambda x: cv2.contourArea(x))
            contours = [largest_cnt]

        cv2.fillPoly(rgb_mask, contours, AIRBAG_COLOR)

    mask = rgb_mask[:, :, 0]
    mask[mask > 0] = 255

    return mask, rgb_mask, contours


def get_seg_ear_from_prediction(class_predict, head_contour):

    ear_mask = np.zeros(class_predict.shape, dtype=np.uint8)
    ear_mask[class_predict == EAR] = 255
    mask = np.zeros(class_predict.shape, dtype=np.uint8)

    if len(head_contour) > 0:
        contours, _ = cv2.findContours(ear_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        head_x, head_y = head_contour[0][:, :, 0], head_contour[0][:, :, 1]
        for cnt in contours:
            x0, y0 = np.mean(cnt, axis=0)[0]
            if np.min(head_x) <= x0 <= np.max(head_x) and np.min(head_y) <= y0 <= np.max(head_y):
                cv2.fillPoly(mask, pts=[cnt], color=[255])
                return mask

    return mask


def partition_by_none(path):
    """
    get sub paths which are separated by 0
    """
    ind = 0
    start = None
    ranges = []
    while ind < len(path):
        if path[ind] != 0 and start is None:
            start = ind
        elif path[ind] == 0 and start is not None:
            end = ind
            ranges.append((start, end))
            start = None
        ind += 1
    if start is not None:
        ranges.append((start, ind))
    return ranges


def main(frame2ab_info='../data_heavy/frame2ab.txt',
         head_masks_info='../data_heavy/head_masks.txt',
         input_images='../data_heavy/frames',
         save_seg_abh='../data_heavy/frames_seg_abh',
         save_seg_abh_vis='../data_heavy/frames_seg_abh_vis',
         save_ear_vis='../data_heavy/frames_ear_only',
         save_ear_coor='../data_heavy/frames_ear_coord_only'):

    os.makedirs(save_seg_abh, exist_ok=True)
    os.makedirs(save_seg_abh_vis, exist_ok=True)
    os.makedirs(save_ear_vis, exist_ok=True)
    os.makedirs(save_ear_coor, exist_ok=True)

    # model loading
    config_file = 'configs/AB__SwinBase__2_local__no_TTA.py'
    checkpoint = 'checkpoints/config_2_16000iter.pth'
    dev = 'cuda:0'

    model = init_segmentor(config_file, checkpoint, device=dev)

    # section Predict
    if not DEBUG:
        file_paths = glob.glob(f"{input_images}/[1|2]-*.png")
    else:
        file_paths = glob.glob(f"{input_images}/1-*.png")

    frame2ab_info_dict = {}
    head_masks_info_dict = {}
    for path in tqdm(file_paths, desc='2D Segmentation:'):
        img_name = os.path.basename(path)
        view = int(img_name.split('-')[0])

        inp_img = cv2.imread(path, 1)
        result = inference_segmentor(model, inp_img)
        pred = result[0]

        head_mask, head_rgb_mask, head_contour = get_seg_head_from_prediction(pred)
        ab_mask, ab_rgb_mask, ab_contours = get_seg_airbag_from_prediction(pred, view=view)

        if view == 2:
            refine = refine_airbag_segmentation(ab_mask, head_mask)
            if refine is not None:
                ab_mask, ab_contours = refine
                ab_rgb_mask = np.zeros_like(ab_rgb_mask)
                ab_rgb_mask[ab_mask == 255] = AIRBAG_COLOR

        head_pixels, head_center, head_rot, x1, y1, x2, y2 = compute_position_and_rotation(
            head_mask, head_rgb_mask, head_contour)
        ab_pixels, ab_center, ab_rot, _, _, _, _ = compute_position_and_rotation(
            ab_mask, ab_rgb_mask, ab_contours)

        dist_x = -1
        dist_y = -1
        if ab_center is not None:
            dist_x = ab_center[0]
            dist_y = ab_center[1]

        head_masks_info_dict[img_name] = [img_name, x1, y1, x2, y2]
        frame2ab_info_dict[img_name] = [img_name, ab_pixels, head_pixels, dist_x, dist_y, head_rot, ab_rot]

        seg_final = merge_2images(head_rgb_mask, ab_rgb_mask, HEAD_COLOR, AIRBAG_COLOR)

        Image.fromarray(seg_final).save(f"{save_seg_abh}/{img_name}")
        blend = cv2.addWeighted(inp_img, 0.3, seg_final, 0.7, 0)
        Image.fromarray(blend).save(f"{save_seg_abh_vis}/{img_name}")

        if view == 1:
            ear_mask = get_seg_ear_from_prediction(pred, head_contour)
            origin_img_copy = inp_img.copy()
            origin_img_copy[ear_mask == 0] *= 0
            cv2.imwrite(f'{save_ear_vis}/{img_name}', origin_img_copy)

            pixels = np.argwhere(ear_mask > 0)
            with open(f'{save_ear_coor}/{img_name}', 'wb') as ear_fp:
                pickle.dump(pixels, ear_fp)

    # using k means to clusters based on AB sizes
    sys.stdin = open("../data_heavy/frames/info.txt", "r")
    frame_indices = [line[:-1] for line in sys.stdin.readlines()]
    airbag_detection_results = []
    for idx in frame_indices:
        im_name = f"1-{idx}.png"
        _, ab_pixels = frame2ab_info_dict[im_name][:2]
        airbag_detection_results.append(ab_pixels)
    ranges = partition_by_none(airbag_detection_results)

    res = kmeans1d.cluster(airbag_detection_results, 2)
    centroids = res.centroids

    # if the small centroid is very small compared to the large centroid, set the cluster to null predictions
    if centroids[0] > 0:
        if len(ranges) > 1 or centroids[1]/centroids[0] >= 5:
            print("unusual airbag detection, trying to handle")
            clusters = res.clusters
            for idx, frame_idx in enumerate(frame_indices):
                im_name = f"1-{frame_idx}.png"
                if clusters[idx] == 0:
                    _, ab_pixels, head_pixels, dist_x, dist_y, head_rot, ab_rot = frame2ab_info_dict[im_name]
                    if ab_pixels == 0:
                        continue
                    frame2ab_info_dict[im_name] = [im_name, 0, head_pixels, dist_x, dist_y, head_rot, ab_rot]
                    print(f" modifying {im_name}: from {ab_pixels} to {0}, resulting key=", frame2ab_info_dict[im_name][:2])

    # writing final predictions into files
    assert len(head_masks_info_dict) == len(frame2ab_info_dict)
    with open(head_masks_info, "w") as fp2:
        with open(frame2ab_info, "w") as fp:
            for img_name in head_masks_info_dict:
                _, ab_pixels, head_pixels, dist_x, dist_y, head_rot, ab_rot = frame2ab_info_dict[img_name]
                _, x1, y1, x2, y2 = head_masks_info_dict[img_name]
                print(img_name, ab_pixels, head_pixels, dist_x, dist_y, head_rot, ab_rot, file=fp)
                print(img_name, x1, y1, x2, y2, file=fp2)


if __name__ == '__main__':
    main()
