import glob
import pickle
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from icecream import ic
import os
from PIL import Image

parser = ArgumentParser()
parser.add_argument('-d', '--debug', type=bool, default=False, help='Debug mode')
args = vars(parser.parse_args())
DEBUG_MODE = args['debug']

# section Runtime Arguments
INP_IMG_DIR     = '../data_heavy/frames'
OUT_SEG_ABH_DIR = '../data_heavy/frames_seg_abh'
OUT_SEG_ABH_VIS_DIR = '../data_heavy/frames_seg_abh_vis'
os.makedirs(OUT_SEG_ABH_DIR, exist_ok=True)
os.makedirs(OUT_SEG_ABH_VIS_DIR, exist_ok=True)

OUT_VIS_EAR_DIR  = '../data_heavy/frames_ear_only'
OUT_EAR_COOR_DIR = '../data_heavy/frames_ear_coord_only'
os.makedirs(OUT_VIS_EAR_DIR, exist_ok=True)
os.makedirs(OUT_EAR_COOR_DIR, exist_ok=True)

CONFIG_FILE = 'configs/AB__SwinBase__2_local__no_TTA.py'
CHECKPOINT  = 'checkpoints/config_2_16000iter.pth'
DEVICE      = 'cuda:0'

BACKGROUND, AIRBAG, SIDE_AIRBAG, HEAD, EAR = 0, 1, 2, 3, 4
HEAD_COLOR = [64, 128, 128]
AIRBAG_COLOR = [192, 128, 128]
EAR_COLOR = [255, 153, 51]


def draw_text_to_image(img, text, pos):
    # Write some Text
    font                   = cv.FONT_HERSHEY_SIMPLEX
    fontScale              = 4
    fontColor              = (255,255,255)
    lineType               = 2

    cv.putText(img,text,
        pos,
        font,
        fontScale,
        fontColor,
        lineType)
    return img


def merge_2images(_im1, _im2, c1, c2):
    output = np.zeros(_im1.shape, dtype=np.uint8)

    indices = np.argwhere(_im1[:, :] == c1)
    output[indices[:, 0], indices[:, 1]] = c1

    indices = np.argwhere(_im2[:, :] == c2)
    output[indices[:, 0], indices[:, 1]] = c2

    return output


def check_contour(_image, _color):
    '''
    Arguments
    ---------
    _image: 3-channel image, background = [0,0,0]
    _color: color to fill the largest object in _image

    Returns
    -------
        _new_image: visualize input image with fillPoly
        center: center of the largest contours
        angle: rotation of the object, in degree
    '''
    _new_image = np.zeros(_image.shape, dtype=np.uint8)

    # gray_img = cv.cvtColor(_image, cv.COLOR_BGR2GRAY)
    cnts, _ = cv.findContours(_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    angle = None
    center = None

    if len(cnts) > 0:
        # Draw cnt for all cnts
        cv.drawContours(_image, cnts, -1, (255, 0, 0), 3)

        # Fill only the largest cnt
        largest_cnt = max(cnts, key=lambda du1: cv.contourArea(du1))
        cv.fillPoly(_new_image, pts=[largest_cnt], color=_color)

        # Compute center of the largest cnt
        ori_rect = cv.minAreaRect(largest_cnt)
        center = (int(ori_rect[0][0]), int(ori_rect[0][1]))

        # Compute cangle of the largest cnt
        oriented_rect = cv.boxPoints(ori_rect)
        oriented_rect = np.int0(oriented_rect)

        ind_by_x = np.argsort(oriented_rect[:, 1])
        top_points = oriented_rect[ind_by_x[:2]]
        bot_points = oriented_rect[ind_by_x[2:]]
        left_top_point = top_points[np.argmax(top_points[:, 0])]
        left_bot_point = bot_points[np.argmax(bot_points[:, 0])]

        x1, y1 = left_top_point
        x2, y2 = left_bot_point

        angle = np.rad2deg(np.arctan2(y2-y1, x1-x2))
        if angle < 0:
            angle += 180

    return _new_image, center, angle


def compute_postion_and_rotation(mask, rgb_mask, contours, vis=False):
    assert mask.shape == rgb_mask.shape[:2]
    if len(contours) < 1:
        return 0, None, None
    elif len(contours) > 1:
        points = contours[0]
        for i in range(1, len(contours)):
            points = np.append(points, contours[i], axis=0)
        hull = cv.convexHull(points)
        copy_mask = np.zeros(rgb_mask.shape, dtype=np.uint8)
        cv.fillPoly(copy_mask, [hull], (255, 255, 255))
        copy_mask = copy_mask[:, :, 0]
        mask = copy_mask

    else:
        hull = contours[0]

    area = np.sum(mask == 255)

    orient_rect = cv.minAreaRect(hull)
    center = (int(orient_rect[0][0]), int(orient_rect[0][1]))

    bb_box = cv.boxPoints(orient_rect)
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

    if vis:
        cv.drawContours(rgb_mask, contours, -1, (255, 0, 0), 2)
        cv.drawContours(rgb_mask, [hull], -1, (0, 255, 0), 2)
        cv.drawContours(rgb_mask, [bb_box], -1, (0, 0, 255), 2)

        text_img = np.zeros(rgb_mask.shape, dtype=np.uint8)
        text_img = draw_text_to_image(text_img, f"angle: {angle}", (10, text_img.shape[0]//2))
        text_img = draw_text_to_image(text_img, f"area: {area}", (10, text_img.shape[0]//2 + 200))
        opencv_show_image('vis', np.hstack([rgb_mask, text_img]))

    return area, center, angle


def opencv_show_image(name, image):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow(name, image)
    cv.waitKey()
    cv.destroyAllWindows()
    return


def get_seg_head_from_prediction(class_predict):
    head_mask = np.zeros(class_predict.shape, dtype=np.uint8)
    head_mask[class_predict == HEAD] = 255
    head_mask[class_predict == EAR] = 255
    rgb_mask = np.zeros((class_predict.shape[0], class_predict.shape[1], 3), dtype=np.uint8)

    contours, _ = cv.findContours(head_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        max_contours = [max(contours, key=lambda x: cv.contourArea(x))]
        cv.fillPoly(rgb_mask, max_contours, HEAD_COLOR)
    else:
        max_contours = contours

    mask = rgb_mask[:, :, 0]
    mask[mask > 0] = 255

    return mask, rgb_mask, max_contours


def get_seg_airbag_from_prediction(class_predict, view=None, kept_area=1000):
    ab_mask = np.zeros(class_predict.shape, dtype=np.uint8)
    ab_mask[class_predict == AIRBAG] = 255
    rgb_mask = np.zeros((class_predict.shape[0], class_predict.shape[1], 3), dtype=np.uint8)

    contours, _ = cv.findContours(ab_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        if view == 2:
            contours = [cnt for cnt in contours if cv.contourArea(cnt) >= kept_area]
        else:
            largest_cnt = max(contours, key=lambda x: cv.contourArea(x))
            contours = [largest_cnt]

        cv.fillPoly(rgb_mask, contours, AIRBAG_COLOR)

    mask = rgb_mask[:, :, 0]
    mask[mask > 0] = 255

    return mask, rgb_mask, contours


def get_seg_ear_from_prediction(class_predict, head_contour):

    ear_mask = np.zeros(class_predict.shape, dtype=np.uint8)
    ear_mask[class_predict == EAR] = 255
    mask = np.zeros(class_predict.shape, dtype=np.uint8)

    if len(head_contour) > 0:
        contours, _ = cv.findContours(ear_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        head_x, head_y = head_contour[0][:, :, 0], head_contour[0][:, :, 1]
        for cnt in contours:
            x0, y0 = np.mean(cnt, axis=0)[0]
            if np.min(head_x) <= x0 <= np.max(head_x) and np.min(head_y) <= y0 <= np.max(head_y):
                cv2.fillPoly(mask, pts=[cnt], color=[255])
                return mask

    return mask


def main(debuging=DEBUG_MODE):
    # Remove previous running results
    previous_result = '../data_heavy/frame2ab.txt'
    if Path(previous_result).exists():
        Path(previous_result).unlink()

    # section Predict
    model = init_segmentor(CONFIG_FILE, CHECKPOINT, device=DEVICE)
    file_paths = glob.glob(f"{INP_IMG_DIR}/*.png")

    with open("../data_heavy/frame2ab.txt", "w") as fp:
        for path in tqdm(file_paths, desc='2D Segmentation:'):
            inp_img = cv.imread(path, 1)
            result    = inference_segmentor(model, inp_img)
            pred    = result[0]

            img_name = os.path.basename(path)
            view = int(img_name.split('-')[0])

            head_mask, head_rgb_mask, head_contour = get_seg_head_from_prediction(pred)
            ab_mask, ab_rgb_mask, ab_contours = get_seg_airbag_from_prediction(pred, view=view)

            head_pixels, head_center, head_rot = compute_postion_and_rotation(head_mask, head_rgb_mask, head_contour, vis=debuging)
            ab_pixels, ab_center, ab_rot = compute_postion_and_rotation(ab_mask, ab_rgb_mask, ab_contours, vis=debuging)

            dist_x = -1
            dist_y = -1
            if ab_center is not None and head_center is not None:
                dist_x = ab_center[0] - head_center[0]
                dist_y = ab_center[1] - head_center[1]

            print(img_name, ab_pixels, head_pixels, dist_x, dist_y, head_rot, ab_rot, file=fp)

            seg_final = merge_2images(head_rgb_mask, ab_rgb_mask, HEAD_COLOR, AIRBAG_COLOR)
            Image.fromarray(seg_final).save(f"{OUT_SEG_ABH_DIR}/{img_name}")
            blend = cv2.addWeighted(inp_img, 0.3, seg_final, 0.7, 0)
            Image.fromarray(blend).save(f"{OUT_SEG_ABH_VIS_DIR}/{img_name}")

            if view == 1:
                ear_mask = get_seg_ear_from_prediction(pred, head_contour)
                orgin_img_copy = inp_img.copy()
                orgin_img_copy[ear_mask == 0] *= 0
                cv.imwrite(f'{OUT_VIS_EAR_DIR}/{img_name}', orgin_img_copy)

                pixels = np.argwhere(ear_mask > 0)
                with open(f'{OUT_EAR_COOR_DIR}/{img_name}', 'wb') as ear_fp:
                    pickle.dump(pixels, ear_fp)


if __name__ == '__main__':
    main()

