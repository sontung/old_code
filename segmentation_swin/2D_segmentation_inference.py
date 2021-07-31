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
CONFIG_FILE = 'configs/AB__SwinBase__2_local__no_TTA.py'
CHECKPOINT = 'checkpoints/config_2_16000iter.pth'
DEVICE = 'cuda:0'

MODEL = init_segmentor(CONFIG_FILE, CHECKPOINT, device=DEVICE)

BACKGROUND, AIRBAG, SIDE_AIRBAG, HEAD, EAR = 0, 1, 2, 3, 4
HEAD_COLOR = [64, 128, 128]
AIRBAG_COLOR = [192, 128, 128]
EAR_COLOR = [255, 153, 51]


def draw_text_to_image(img, text, pos):
    # Write some Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 4
    font_color = (255, 255, 255)
    line_type = 2

    cv2.putText(img,
               text,
               pos,
               font,
               font_scale,
               font_color,
               line_type)
    return img


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


def segment_all_video(root='/media/hblab/01D5F2DD5173DEA0/AirBag/3d-air-bag-p2/data_video'):
    frames = 'all_frames'
    segs = 'all_segments'

    sub_frames_folder = os.walk(os.path.join(root, frames)).__next__()[1]
    sub_segs_folder = os.walk(os.path.join(root, segs)).__next__()[1]

    for f in tqdm(sub_frames_folder, desc="Segmentation"):
        if f in sub_segs_folder:
            print(f"Skipping {f}")
        else:
            directory = os.path.join(root, segs, f)
            os.makedirs(directory, exist_ok=True)
            main(frame2ab_info=os.path.join(directory, 'frame2ab.txt'),
                 input_images=os.path.join(root, frames, f),
                 save_seg_abh=os.path.join(directory, "frames_seg_abh"),
                 save_seg_abh_vis=os.path.join(directory, "frames_seg_abh_vis"),
                 save_ear_vis=os.path.join(directory, "frames_ear_only"),
                 save_ear_coor=os.path.join(directory, "frames_ear_coord_only"))

    return


def compute_2d_x_axis(ab_contours, head_contour, origin_img, img_name, debuging=DEBUG_MODE):

    def bb_box(contour):
        top_left = np.min(contour, axis=0)[0]
        bottom_right = np.max(contour, axis=0)[0]
        center = ((top_left[0] + bottom_right[0])/2, (top_left[1] + bottom_right[1])/2)
        return top_left, bottom_right, center

    def draw_rectange(_image, top_left, bottom_right, center):
        cv2.rectangle(_image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.circle(_image, center, 2, (0, 255, 0), 2)
        return _image

    def float2int(rect):
        tl, br, center = rect
        tl = (int(tl[0]), int(tl[1]))
        br = (int(br[0]), int(br[1]))
        center = (int(center[0]), int(center[1]))
        return tl, br, center

    if len(ab_contours) < 1 or len(head_contour) < 1:
        return None

    if len(ab_contours) == 1:
        new_ab_contour = ab_contours
    else:
        points = ab_contours[0]
        for i in range(len(ab_contours)):
            points = np.append(points, ab_contours[i], axis=0)

        new_ab_contour = [cv2.convexHull(points)]

    ab_rect = bb_box(new_ab_contour[0])
    head_rect = bb_box(head_contour[0])

    if debuging:
        os.makedirs('debugs', exist_ok=True)
        mask = np.zeros(origin_img.shape, dtype=np.uint8)
        cv2.fillPoly(mask, pts=new_ab_contour, color=AIRBAG_COLOR)
        cv2.fillPoly(mask, pts=head_contour, color=HEAD_COLOR)

        blend = cv2.addWeighted(origin_img, 0.3, mask, 0.7, 0)

        ab_rect = float2int(ab_rect)
        blend = draw_rectange(blend, ab_rect[0], ab_rect[1], ab_rect[2])

        head_rect = float2int(head_rect)
        blend = draw_rectange(blend, head_rect[0], head_rect[1], head_rect[2])

        Image.fromarray(blend).save(f'debugs/{img_name}')

    return


def partition_by_none(path):
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

    # section Predict
    file_paths = glob.glob(f"{input_images}/[1|2]-*.png")
    frame2ab_info_dict = {}
    head_masks_info_dict = {}

    for path in tqdm(file_paths, desc='2D Segmentation:'):
        img_name = os.path.basename(path)
        view = int(img_name.split('-')[0])

        inp_img = cv2.imread(path, 1)
        result = inference_segmentor(MODEL, inp_img)
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

    # frame2ab_info_dict = {'2-15.png': ['2-15.png', 0, 96211, -1, -1, 87.70938995736148, None],
    #  '2-46.png': ['2-46.png', 158841, 30728, 684, 250, 94.35107795157388, 68.4286928087454],
    #  '2-8.png': ['2-8.png', 0, 96279, -1, -1, 87.95914313006675, None],
    #  '2-76.png': ['2-76.png', 137539, 80739, 804, 295, 87.59640296220643, 77.44244378372112],
    #  '1-26.png': ['1-26.png', 5645, 17081, 633, 410, 103.3924977537511, 147.07213734683586],
    #  '2-36.png': ['2-36.png', 138012, 44325, 723, 280, 86.94711748520757, 79.13594007791228],
    #  '2-50.png': ['2-50.png', 167116, 31105, 679, 299, 115.21349587147864, 79.94566440000565],
    #  '1-48.png': ['1-48.png', 57833, 0, 510, 159, None, 63.07685486359844],
    #  '2-65.png': ['2-65.png', 168680, 66823, 737, 293, 83.17525357560896, 89.7632419280929],
    #  '1-39.png': ['1-39.png', 59584, 0, 513, 150, None, 50.64331828053398],
    #  '1-6.png': ['1-6.png', 1682, 17026, 213, 183, 104.69196052723488, 99.46232220802563],
    #  '1-40.png': ['1-40.png', 60883, 0, 505, 149, None, 57.102598294906635],
    #  '2-57.png': ['2-57.png', 177409, 53697, 690, 284, 90.0, 84.59089182394881],
    #  '2-72.png': ['2-72.png', 141574, 77273, 776, 298, 86.24354745219593, 82.36271315944244],
    #  '2-16.png': ['2-16.png', 0, 96213, -1, -1, 87.7042938282712, None],
    #  '2-33.png': ['2-33.png', 137879, 57547, 754, 326, 90.0, 90.0],
    #  '1-65.png': ['1-65.png', 54672, 12638, 517, 187, 77.71904559490062, 76.8241062992028],
    #  '1-81.png': ['1-81.png', 73101, 15694, 513, 209, 93.90854429445946, 74.12120125565627],
    #  '2-6.png': ['2-6.png', 0, 96363, -1, -1, 87.58228088486815, None],
    #  '1-21.png': ['1-21.png', 1533, 17054, 212, 186, 103.5479389880005, 88.45184230102204],
    #  '1-47.png': ['1-47.png', 58688, 0, 510, 154, None, 61.50436138175502],
    #  '1-61.png': ['1-61.png', 51454, 10872, 523, 185, 73.26400414852392, 73.69383260883696],
    #  '1-38.png': ['1-38.png', 59032, 0, 526, 154, None, 41.43084284900533],
    #  '2-58.png': ['2-58.png', 176139, 54456, 700, 276, 90.0, 84.74619724873774],
    #  '2-7.png': ['2-7.png', 0, 96250, -1, -1, 88.59971536079435, None],
    #  '1-78.png': ['1-78.png', 69831, 14991, 507, 202, 89.27477570094075, 72.5790298998633],
    #  '2-49.png': ['2-49.png', 163773, 31222, 690, 272, 54.833563964207116, 71.91655461695134],
    #  '1-28.png': ['1-28.png', 0, 16528, -1, -1, 102.78907227632627, None],
    #  '2-27.png': ['2-27.png', 0, 88834, -1, -1, 88.93908830973577, None],
    #  '1-74.png': ['1-74.png', 64384, 13650, 493, 194, 83.74096369487775, 67.22962109123779],
    #  '2-47.png': ['2-47.png', 157296, 32612, 690, 241, 95.27389595735177, 73.07248693585295],
    #  '2-45.png': ['2-45.png', 159678, 28500, 698, 241, 55.00797980144134, 67.26518835688593],
    #  '1-14.png': ['1-14.png', 1769, 17007, 208, 181, 103.29857033049429, 108.94650468950906],
    #  '2-53.png': ['2-53.png', 171361, 45695, 670, 288, 53.65872123236698, 79.52257640399513],
    #  '1-35.png': ['1-35.png', 58896, 944, 480, 160, 48.50353164478446, 48.179830119864235],
    #  '1-59.png': ['1-59.png', 51584, 8552, 514, 180, 72.92603397005406, 65.06307266608825],
    #  '2-59.png': ['2-59.png', 177005, 55855, 708, 278, 86.77309036771332, 84.82954422036961],
    #  '1-31.png': ['1-31.png', 38651, 15726, 505, 152, 94.96974072811031, 38.50758815656127],
    #  '1-33.png': ['1-33.png', 49139, 12479, 514, 158, 61.76808174133644, 49.62923494779759],
    #  '1-7.png': ['1-7.png', 2190, 17015, 213, 187, 104.60673735444699, 89.24615166692924],
    #  '1-80.png': ['1-80.png', 72052, 15569, 511, 207, 88.56790381583535, 73.61045966596522],
    #  '2-17.png': ['2-17.png', 0, 96249, -1, -1, 87.06758996870961, None],
    #  '2-54.png': ['2-54.png', 172613, 47943, 676, 262, 99.80842100482705, 84.36573334423261],
    #  '2-35.png': ['2-35.png', 136039, 47608, 736, 310, 83.1643538145388, 90.0],
    #  '2-74.png': ['2-74.png', 146604, 79185, 790, 291, 88.8542371618249, 79.90249561592472],
    #  '1-50.png': ['1-50.png', 55136, 0, 511, 170, None, 62.775563592840015],
    #  '2-56.png': ['2-56.png', 176741, 52941, 679, 284, 97.75725005004338, 82.32825410384446],
    #  '1-8.png': ['1-8.png', 1955, 17001, 215, 185, 104.28109573597084, 90.0],
    #  '1-44.png': ['1-44.png', 60127, 0, 508, 155, None, 61.16137250672043],
    #  '1-53.png': ['1-53.png', 53175, 968, 511, 176, 58.6097318958122, 66.47500333548889],
    #  '2-14.png': ['2-14.png', 0, 96156, -1, -1, 87.83652159285387, None],
    #  '2-71.png': ['2-71.png', 144869, 75855, 771, 294, 83.77155803063656, 83.08108268848233],
    #  '2-13.png': ['2-13.png', 0, 96171, -1, -1, 87.70938995736148, None],
    #  '1-79.png': ['1-79.png', 70993, 15250, 509, 205, 90.0, 73.19291226426981],
    #  '1-58.png': ['1-58.png', 51702, 7266, 515, 179, 73.1873737645758, 66.56113122483765],
    #  '2-43.png': ['2-43.png', 177082, 31179, 727, 269, 56.46395301569255, 66.24073473991673],
    #  '1-60.png': ['1-60.png', 51368, 9692, 519, 182, 72.41651812223354, 68.96248897457819],
    #  '2-22.png': ['2-22.png', 0, 95501, -1, -1, 86.79887297385115, None],
    #  '2-19.png': ['2-19.png', 0, 96211, -1, -1, 88.08659871424813, None],
    #  '1-41.png': ['1-41.png', 61399, 0, 507, 151, None, 58.9283419391537],
    #  '2-11.png': ['2-11.png', 0, 96127, -1, -1, 87.83652159285387, None],
    #  '1-15.png': ['1-15.png', 1866, 17015, 210, 181, 103.29857033049429, 106.92751306414705],
    #  '1-71.png': ['1-71.png', 61176, 13432, 509, 195, 81.02737338510362, 75.96375653207353],
    #  '1-67.png': ['1-67.png', 56617, 12995, 513, 190, 79.72899627952093, 75.37912601136834],
    #  '2-4.png': ['2-4.png', 0, 96522, -1, -1, 87.84131411913795, None],
    #  '2-25.png': ['2-25.png', 0, 91851, -1, -1, 87.63592041408052, None],
    #  '2-41.png': ['2-41.png', 162330, 32887, 746, 324, 87.81835859644853, 115.17753542963479],
    #  '2-32.png': ['2-32.png', 115561, 64211, 779, 330, 89.34521959773231, 93.73762954341011],
    #  '1-68.png': ['1-68.png', 57417, 13199, 513, 192, 80.10174005584747, 75.76966588047844],
    #  '1-69.png': ['1-69.png', 58643, 13428, 512, 192, 79.79602627826831, 76.11881915623661],
    #  '2-52.png': ['2-52.png', 168737, 40177, 671, 297, 54.18208719292104, 80.75091007701604],
    #  '2-64.png': ['2-64.png', 171065, 65565, 734, 296, 84.01504768488459, 90.0],
    #  '2-10.png': ['2-10.png', 0, 96166, -1, -1, 87.83170774950469, None],
    #  '1-22.png': ['1-22.png', 586, 16928, 205, 218, 104.60673735444699, 49.9392155421262],
    #  '1-12.png': ['1-12.png', 706, 17028, 212, 192, 103.62699485989154, 79.26110289909457],
    #  '1-25.png': ['1-25.png', 6266, 17271, 175, 172, 103.79380861217655, 47.060111023723124],
    #  '1-63.png': ['1-63.png', 52824, 12000, 522, 186, 77.93223104134667, 76.122470196792],
    #  '1-64.png': ['1-64.png', 53653, 12460, 520, 187, 77.34744349944204, 76.51693997259474],
    #  '2-67.png': ['2-67.png', 160655, 70401, 748, 294, 81.43285150218546, 87.34819582952302],
    #  '2-37.png': ['2-37.png', 145289, 41545, 714, 299, 90.0, 77.12108960020304],
    #  '2-63.png': ['2-63.png', 173206, 63402, 729, 290, 84.38241940987318, 90.0],
    #  '2-23.png': ['2-23.png', 0, 94233, -1, -1, 87.40919801480489, None],
    #  '1-13.png': ['1-13.png', 1173, 16996, 211, 185, 103.62699485989154, 90.0],
    #  '2-77.png': ['2-77.png', 135771, 80658, 803, 293, 87.5904604682114, 78.74168533139623],
    #  '1-73.png': ['1-73.png', 63167, 13565, 492, 192, 82.55258314980485, 67.35518044573564],
    #  '2-24.png': ['2-24.png', 0, 93373, -1, -1, 87.78739995660182, None],
    #  '2-38.png': ['2-38.png', 145471, 38445, 711, 323, 88.53119928561418, 90.0],
    #  '1-37.png': ['1-37.png', 58072, 0, 503, 150, None, 37.498043171501074],
    #  '1-23.png': ['1-23.png', 6, 16938, 244, 170, 104.93141717813755, 135.0],
    #  '1-10.png': ['1-10.png', 906, 17001, 213, 183, 103.95450917313686, 93.31778116833485],
    #  '1-27.png': ['1-27.png', 0, 16783, -1, -1, 104.03624346792648, None],
    #  '2-69.png': ['2-69.png', 153191, 72988, 761, 294, 80.10939505824196, 84.63642442901988],
    #  '2-61.png': ['2-61.png', 175972, 60030, 717, 272, 83.96543078742906, 81.8327649358691],
    #  '1-76.png': ['1-76.png', 67002, 14342, 501, 199, 87.08299271337229, 70.84761022059627],
    #  '2-66.png': ['2-66.png', 163848, 68226, 743, 281, 82.05996772473462, 88.91680550468602],
    #  '1-72.png': ['1-72.png', 62168, 13451, 495, 191, 81.8174770021786, 67.56700907787462],
    #  '1-29.png': ['1-29.png', 13262, 16192, 537, 180, 99.46232220802563, 39.13845860180353],
    #  '2-12.png': ['2-12.png', 0, 96122, -1, -1, 87.83170774950469, None],
    #  '2-42.png': ['2-42.png', 164455, 31442, 747, 288, 90.0, 53.3516412385638],
    #  '2-18.png': ['2-18.png', 0, 96263, -1, -1, 87.83170774950469, None],
    #  '2-20.png': ['2-20.png', 0, 96207, -1, -1, 87.4551956201869, None],
    #  '2-44.png': ['2-44.png', 163306, 29442, 712, 261, 55.74585722957785, 66.21587374754871],
    #  '1-18.png': ['1-18.png', 1968, 17030, 213, 182, 103.29857033049429, 100.37584492005105],
    #  '2-26.png': ['2-26.png', 0, 89962, -1, -1, 88.41253253517942, None],
    #  '2-2.png': ['2-2.png', 0, 96421, -1, -1, 88.34524882351319, None],
    #  '1-17.png': ['1-17.png', 143, 17025, 213, 203, 103.29857033049429, 85.60129464500447],
    #  '2-48.png': ['2-48.png', 159536, 30900, 691, 234, 54.80842100482706, 77.02657247468889],
    #  '1-9.png': ['1-9.png', 763, 16996, 213, 192, 104.28109573597084, 82.74680538727468],
    #  '1-5.png': ['1-5.png', 1995, 17016, 213, 184, 104.28109573597084, 94.63546342690265],
    #  '1-32.png': ['1-32.png', 46689, 13990, 505, 152, 90.0, 39.076598522385595],
    #  '1-55.png': ['1-55.png', 52280, 3344, 508, 174, 74.86416449625196, 63.58204934570755],
    #  '2-34.png': ['2-34.png', 132621, 51194, 739, 312, 90.0, 90.0],
    #  '2-73.png': ['2-73.png', 150216, 78476, 784, 291, 90.0, 81.08756503047726],
    #  '1-24.png': ['1-24.png', 5411, 17158, 196, 176, 104.1169416955696, 118.21735585472932],
    #  '2-21.png': ['2-21.png', 0, 96231, -1, -1, 88.60281897270363, None],
    #  '1-77.png': ['1-77.png', 68225, 14628, 503, 201, 88.54978370906657, 71.15433577831291],
    #  '2-70.png': ['2-70.png', 148876, 74257, 767, 292, 78.54887564935323, 83.18721914064756],
    #  '2-68.png': ['2-68.png', 157894, 72236, 754, 298, 81.20945681718605, 85.99457943757857],
    #  '1-20.png': ['1-20.png', 24, 17000, 214, 208, 103.5479389880005, 90.0],
    #  '2-60.png': ['2-60.png', 175977, 57904, 712, 272, 84.9868862449642, 82.69424046668918],
    #  '1-56.png': ['1-56.png', 51718, 4787, 511, 174, 73.65711745655814, 64.97321258666538],
    #  '1-1.png': ['1-1.png', 1634, 17049, 214, 180, 103.95450917313686, 103.65041913475699],
    #  '2-9.png': ['2-9.png', 0, 96246, -1, -1, 87.83170774950469, None],
    #  '1-66.png': ['1-66.png', 55548, 12811, 514, 188, 78.61613752892787, 75.69054984484981],
    #  '1-45.png': ['1-45.png', 59236, 0, 509, 152, None, 60.235409113359445],
    #  '2-3.png': ['2-3.png', 0, 96397, -1, -1, 88.85677698094257, None],
    #  '2-39.png': ['2-39.png', 166271, 34954, 711, 314, 87.26284369493932, 90.0],
    #  '2-5.png': ['2-5.png', 0, 96445, -1, -1, 87.84131411913795, None],
    #  '1-34.png': ['1-34.png', 58347, 4164, 490, 167, 63.749757630311116, 48.57633437499735],
    #  '1-43.png': ['1-43.png', 61208, 0, 507, 155, None, 61.1134182330893],
    #  '1-52.png': ['1-52.png', 53817, 233, 508, 175, 90.0, 64.76373635309642],
    #  '2-30.png': ['2-30.png', 33901, 74420, 688, 300, 86.66908261954995, 104.58714744714506],
    #  '2-1.png': ['2-1.png', 0, 96341, -1, -1, 87.96367454520173, None],
    #  '1-62.png': ['1-62.png', 51997, 11514, 524, 186, 78.69006752597979, 75.88401297950885],
    #  '1-2.png': ['1-2.png', 948, 17012, 218, 181, 104.28109573597084, 96.80905017961341],
    #  '1-4.png': ['1-4.png', 1623, 16991, 213, 186, 104.28109573597084, 89.21517539700811],
    #  '1-42.png': ['1-42.png', 61268, 0, 509, 157, None, 61.1134182330893],
    #  '1-30.png': ['1-30.png', 35943, 15753, 524, 144, 98.88065915052024, 122.42453450025559],
    #  '1-11.png': ['1-11.png', 1697, 17022, 212, 185, 103.95450917313686, 90.0],
    #  '1-54.png': ['1-54.png', 52819, 1970, 508, 178, 64.11972632993115, 65.90177949487497],
    #  '1-70.png': ['1-70.png', 60150, 13290, 502, 192, 80.53767779197439, 71.61562113703845],
    #  '1-57.png': ['1-57.png', 51492, 6051, 511, 180, 73.72373705794004, 65.3891647802817],
    #  '1-75.png': ['1-75.png', 65672, 13969, 497, 197, 85.94065363520075, 69.19625258779125],
    #  '2-62.png': ['2-62.png', 175366, 62102, 722, 280, 83.46665741524556, 81.22407885186954],
    #  '2-29.png': ['2-29.png', 13256, 80385, 687, 397, 86.60028584087674, 83.6598082540901],
    #  '2-31.png': ['2-31.png', 104489, 68382, 718, 294, 88.87983420240175, 105.53791887201729],
    #  '1-36.png': ['1-36.png', 59589, 0, 478, 151, None, 45.11482104772024],
    #  '1-19.png': ['1-19.png', 1188, 17007, 212, 184, 103.62699485989154, 92.38594403038883],
    #  '2-51.png': ['2-51.png', 167562, 35513, 675, 297, 55.33450266669379, 80.32073153791983],
    #  '1-46.png': ['1-46.png', 59139, 0, 510, 153, None, 60.94539590092286],
    #  '1-49.png': ['1-49.png', 56503, 0, 512, 165, None, 62.20374992727998],
    #  '1-3.png': ['1-3.png', 1610, 16982, 215, 184, 104.36458244969721, 94.69868051729944],
    #  '2-40.png': ['2-40.png', 156990, 32208, 735, 331, 86.76498494221543, 111.16809947055168],
    #  '1-16.png': ['1-16.png', 263, 17014, 211, 202, 103.29857033049429, 77.66091272167381],
    #  '2-28.png': ['2-28.png', 0, 86116, -1, -1, 87.29299568160015, None],
    #  '2-75.png': ['2-75.png', 141775, 80144, 800, 293, 73.77263895648451, 77.51362312449864],
    #  '1-51.png': ['1-51.png', 54289, 0, 508, 173, None, 62.91811946912492],
    #  '2-55.png': ['2-55.png', 174144, 50199, 677, 259, 101.74563342528795, 83.14698056109897]}
    # head_masks_info_dict = {'2-15.png': ['2-15.png', 728, 20, 710, 470], '2-46.png': ['2-46.png', 528, 214, 542, 398], '2-8.png': ['2-8.png', 730, 20, 714, 469], '2-76.png': ['2-76.png', 643, 76, 626, 481], '1-26.png': ['1-26.png', 334, 2, 374, 170], '2-36.png': ['2-36.png', 561, 154, 545, 454], '2-50.png': ['2-50.png', 438, 210, 543, 433], '1-48.png': ['1-48.png', 0, 0, 0, 0], '2-65.png': ['2-65.png', 564, 112, 519, 488], '1-39.png': ['1-39.png', 0, 0, 0, 0], '1-6.png': ['1-6.png', 330, 6, 373, 170], '1-40.png': ['1-40.png', 0, 0, 0, 0], '2-57.png': ['2-57.png', 501, 152, 501, 487], '2-72.png': ['2-72.png', 612, 82, 586, 478], '2-16.png': ['2-16.png', 728, 20, 710, 469], '2-33.png': ['2-33.png', 619, 155, 619, 492], '1-65.png': ['1-65.png', 458, 72, 426, 219], '1-81.png': ['1-81.png', 411, 39, 422, 200], '2-6.png': ['2-6.png', 733, 20, 714, 470], '1-21.png': ['1-21.png', 333, 6, 373, 172], '1-47.png': ['1-47.png', 0, 0, 0, 0], '1-61.png': ['1-61.png', 457, 82, 414, 225], '1-38.png': ['1-38.png', 0, 0, 0, 0], '2-58.png': ['2-58.png', 508, 140, 508, 472], '2-7.png': ['2-7.png', 730, 18, 719, 468], '1-78.png': ['1-78.png', 419, 45, 417, 203], '2-49.png': ['2-49.png', 580, 334, 456, 510], '1-28.png': ['1-28.png', 350, 21, 387, 184], '2-27.png': ['2-27.png', 706, 36, 698, 468], '1-74.png': ['1-74.png', 431, 53, 414, 208], '2-47.png': ['2-47.png', 522, 201, 540, 396], '2-45.png': ['2-45.png', 576, 321, 464, 481], '1-14.png': ['1-14.png', 334, 7, 373, 172], '2-53.png': ['2-53.png', 601, 310, 434, 537], '1-35.png': ['1-35.png', 431, 59, 385, 111], '1-59.png': ['1-59.png', 452, 84, 409, 224], '2-59.png': ['2-59.png', 520, 144, 501, 481], '1-31.png': ['1-31.png', 422, 38, 436, 199], '1-33.png': ['1-33.png', 463, 70, 383, 219], '1-7.png': ['1-7.png', 330, 6, 373, 171], '1-80.png': ['1-80.png', 418, 45, 414, 205], '2-17.png': ['2-17.png', 730, 21, 707, 470], '2-54.png': ['2-54.png', 474, 128, 525, 423], '2-35.png': ['2-35.png', 591, 162, 553, 479], '2-74.png': ['2-74.png', 623, 70, 615, 470], '1-50.png': ['1-50.png', 0, 0, 0, 0], '2-56.png': ['2-56.png', 476, 143, 520, 466], '1-8.png': ['1-8.png', 331, 6, 373, 171], '1-44.png': ['1-44.png', 0, 0, 0, 0], '1-53.png': ['1-53.png', 432, 159, 396, 218], '2-14.png': ['2-14.png', 728, 20, 711, 470], '2-71.png': ['2-71.png', 610, 88, 567, 482], '2-13.png': ['2-13.png', 728, 20, 710, 470], '1-79.png': ['1-79.png', 418, 45, 418, 204], '1-58.png': ['1-58.png', 450, 85, 408, 224], '2-43.png': ['2-43.png', 606, 319, 492, 491], '1-60.png': ['1-60.png', 456, 83, 411, 225], '2-22.png': ['2-22.png', 733, 33, 708, 480], '2-19.png': ['2-19.png', 729, 19, 714, 468], '1-41.png': ['1-41.png', 0, 0, 0, 0], '2-11.png': ['2-11.png', 729, 20, 712, 470], '1-15.png': ['1-15.png', 334, 7, 373, 172], '1-71.png': ['1-71.png', 443, 59, 419, 211], '1-67.png': ['1-67.png', 454, 66, 427, 215], '2-4.png': ['2-4.png', 733, 19, 716, 470], '2-25.png': ['2-25.png', 724, 19, 706, 455], '2-41.png': ['2-41.png', 555, 240, 547, 450], '2-32.png': ['2-32.png', 635, 138, 631, 488], '1-68.png': ['1-68.png', 456, 66, 430, 215], '1-69.png': ['1-69.png', 458, 64, 431, 214], '2-52.png': ['2-52.png', 594, 319, 441, 531], '2-64.png': ['2-64.png', 556, 117, 517, 489], '2-10.png': ['2-10.png', 730, 20, 713, 469], '1-22.png': ['1-22.png', 336, 7, 379, 172], '1-12.png': ['1-12.png', 333, 7, 373, 172], '1-25.png': ['1-25.png', 329, -1, 370, 166], '1-63.png': ['1-63.png', 456, 75, 425, 220], '1-64.png': ['1-64.png', 459, 74, 426, 221], '2-67.png': ['2-67.png', 585, 108, 527, 493], '2-37.png': ['2-37.png', 538, 171, 538, 460], '2-63.png': ['2-63.png', 550, 119, 514, 485], '2-23.png': ['2-23.png', 734, 24, 714, 466], '1-13.png': ['1-13.png', 333, 7, 373, 172], '2-77.png': ['2-77.png', 646, 68, 629, 472], '1-73.png': ['1-73.png', 437, 56, 417, 209], '2-24.png': ['2-24.png', 731, 27, 714, 467], '2-38.png': ['2-38.png', 538, 206, 531, 479], '1-37.png': ['1-37.png', 0, 0, 0, 0], '1-23.png': ['1-23.png', 338, 11, 382, 176], '1-10.png': ['1-10.png', 332, 7, 373, 172], '1-27.png': ['1-27.png', 343, 20, 384, 184], '2-69.png': ['2-69.png', 603, 105, 535, 495], '2-61.png': ['2-61.png', 537, 127, 500, 477], '1-76.png': ['1-76.png', 424, 48, 416, 205], '2-66.png': ['2-66.png', 577, 101, 524, 481], '1-72.png': ['1-72.png', 441, 58, 419, 211], '1-29.png': ['1-29.png', 370, 23, 397, 185], '2-12.png': ['2-12.png', 728, 20, 711, 469], '2-42.png': ['2-42.png', 563, 225, 563, 420], '2-18.png': ['2-18.png', 729, 20, 712, 469], '2-20.png': ['2-20.png', 731, 20, 711, 470], '2-44.png': ['2-44.png', 590, 331, 479, 494], '1-18.png': ['1-18.png', 334, 7, 373, 172], '2-26.png': ['2-26.png', 717, 7, 705, 440], '2-2.png': ['2-2.png', 732, 18, 719, 468], '1-17.png': ['1-17.png', 334, 7, 373, 172], '2-48.png': ['2-48.png', 579, 314, 457, 487], '1-9.png': ['1-9.png', 331, 6, 373, 171], '1-5.png': ['1-5.png', 331, 6, 373, 171], '1-32.png': ['1-32.png', 431, 44, 431, 203], '1-55.png': ['1-55.png', 435, 97, 402, 219], '2-34.png': ['2-34.png', 602, 154, 602, 478], '2-73.png': ['2-73.png', 610, 68, 610, 466], '1-24.png': ['1-24.png', 332, 2, 374, 169], '2-21.png': ['2-21.png', 729, 18, 718, 469], '1-77.png': ['1-77.png', 421, 46, 417, 204], '2-70.png': ['2-70.png', 614, 102, 535, 492], '2-68.png': ['2-68.png', 591, 112, 531, 500], '1-20.png': ['1-20.png', 333, 6, 373, 172], '2-60.png': ['2-60.png', 529, 135, 499, 477], '1-56.png': ['1-56.png', 443, 88, 404, 221], '1-1.png': ['1-1.png', 332, 6, 373, 171], '2-9.png': ['2-9.png', 730, 20, 713, 469], '1-66.png': ['1-66.png', 455, 69, 425, 218], '1-45.png': ['1-45.png', 0, 0, 0, 0], '2-3.png': ['2-3.png', 731, 17, 722, 468], '2-39.png': ['2-39.png', 546, 215, 534, 466], '2-5.png': ['2-5.png', 732, 19, 715, 470], '1-34.png': ['1-34.png', 436, 62, 364, 208], '1-43.png': ['1-43.png', 0, 0, 0, 0], '1-52.png': ['1-52.png', 401, 187, 401, 208], '2-30.png': ['2-30.png', 664, 126, 642, 504], '2-1.png': ['2-1.png', 733, 20, 717, 470], '1-62.png': ['1-62.png', 455, 76, 426, 221], '1-2.png': ['1-2.png', 331, 6, 373, 171], '1-4.png': ['1-4.png', 331, 6, 373, 171], '1-42.png': ['1-42.png', 0, 0, 0, 0], '1-30.png': ['1-30.png', 395, 34, 420, 194], '1-11.png': ['1-11.png', 332, 7, 373, 172], '1-54.png': ['1-54.png', 447, 119, 398, 220], '1-70.png': ['1-70.png', 446, 61, 421, 211], '1-57.png': ['1-57.png', 447, 87, 407, 224], '1-75.png': ['1-75.png', 426, 50, 415, 205], '2-62.png': ['2-62.png', 545, 129, 504, 487], '2-29.png': ['2-29.png', 682, 114, 658, 518], '2-31.png': ['2-31.png', 645, 134, 638, 492], '1-36.png': ['1-36.png', 0, 0, 0, 0], '1-19.png': ['1-19.png', 333, 7, 373, 172], '2-51.png': ['2-51.png', 590, 325, 451, 526], '1-46.png': ['1-46.png', 0, 0, 0, 0], '1-49.png': ['1-49.png', 0, 0, 0, 0], '1-3.png': ['1-3.png', 331, 6, 373, 170], '2-40.png': ['2-40.png', 548, 234, 535, 464], '1-16.png': ['1-16.png', 334, 7, 373, 172], '2-28.png': ['2-28.png', 697, 67, 677, 490], '2-75.png': ['2-75.png', 667, 115, 550, 517], '1-51.png': ['1-51.png', 0, 0, 0, 0], '2-55.png': ['2-55.png', 470, 120, 533, 423]}

    sys.stdin = open("../data_heavy/frames/info.txt", "r")
    frame_indices = [line[:-1] for line in sys.stdin.readlines()]
    airbag_detection_results = []
    for idx in frame_indices:
        im_name = f"1-{idx}.png"
        _, ab_pixels = frame2ab_info_dict[im_name][:2]
        airbag_detection_results.append(ab_pixels)
    ranges = partition_by_none(airbag_detection_results)
    if len(ranges) > 1:
        print("unusual airbag detection, trying to handle")
        res = kmeans1d.cluster(airbag_detection_results, 2)
        clusters = res.clusters
        for idx, frame_idx in enumerate(frame_indices):
            im_name = f"1-{frame_idx}.png"
            if clusters[idx] == 0:
                _, ab_pixels, head_pixels, dist_x, dist_y, head_rot, ab_rot = frame2ab_info_dict[im_name]
                if ab_pixels == 0:
                    continue
                frame2ab_info_dict[im_name] = [im_name, 0, head_pixels, dist_x, dist_y, head_rot, ab_rot]
                print(f" modifying {im_name}: from {ab_pixels} to {0}, resulting key=", frame2ab_info_dict[im_name][:2])

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
