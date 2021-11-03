import sys
import json
import cv2
import numpy as np
import os
import open3d as o3d
import pickle
import kmeans1d
from os import listdir
from os.path import isfile, join
from solve_position import visualize, visualize_wo_ab
from test_model import new_model
from scipy.spatial.transform import Rotation as rot_mat_compute


def keep_one_biggest_contour(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cnts, _ = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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


BACKGROUND, AIRBAG, SIDE_AIRBAG, HEAD, EAR = 0, 1, 2, 3, 4
HEAD_COLOR = [64, 128, 128]
AIRBAG_COLOR = [192, 128, 128]
EAR_COLOR = [255, 153, 51]
DEBUG = False


def merge_2images(_im1, _im2, c1, c2):
    """
    merge two visualizations into one matrix
    """
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


def read_json(name, img):
    color2id = {"head": 3, "ab": 1, "ear": 4}
    res = {}
    try:
        with open(name, 'r') as myfile:
            data = myfile.read()
    except FileNotFoundError:
        return res
    obj = json.loads(data)
    pred = np.zeros_like(img, dtype=np.uint8)
    view = name.split(".mp4.json")[0].split("-")[-1]

    for shape in obj["shapes"]:
        cv2.fillPoly(pred, pts=[np.array(shape["points"], dtype=np.int32)], color=(color2id[shape["label"]], 0, 0))
    pred = pred[:, :, 0]
    head_mask, head_rgb_mask, head_contour = get_seg_head_from_prediction(pred)
    ab_mask, ab_rgb_mask, ab_contours = get_seg_airbag_from_prediction(pred, view=view)
    head_pixels, head_center, head_rot, x1, y1, x2, y2 = compute_position_and_rotation(
        head_mask, head_rgb_mask, head_contour)
    ab_pixels, ab_center, ab_rot, _, _, _, _ = compute_position_and_rotation(
        ab_mask, ab_rgb_mask, ab_contours)

    res["head center"] = head_center
    res["ab center"] = ab_center
    if res["ab center"] is not None:
        res["ab center"] = np.array(ab_center)
    if res["head center"] is not None:
        res["head center"] = np.array(head_center)
    res["head rot"] = head_rot
    res["ab rot"] = ab_rot
    res["ab center"] = np.array(ab_center)
    res["head pixels"] = head_pixels
    res["ab pixels"] = ab_pixels

    return res


def process_3d(comp):
    pcd = new_model()
    pcd.translate([0, 0, 0], relative=False)
    pcd.compute_vertex_normals()
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.get_view_control().set_zoom(1.5)

    if "1" in comp:
        rot1 = comp["1"][0]["head rot"]
        if rot1 > 90:
            rot_mat2 = rot_mat_compute.from_euler('x', -rot1+90,
                                                  degrees=True).as_matrix()
        else:
            rot_mat2 = rot_mat_compute.from_euler('x', rot1-90,
                                                  degrees=True).as_matrix()

        pcd.rotate(rot_mat2, pcd.get_center())
    if "2" in comp:
        rot2 = comp["2"][0]["head rot"]
        rot_mat = rot_mat_compute.from_euler('z', rot2-90,
                                             degrees=True).as_matrix()
        pcd.rotate(rot_mat, pcd.get_center())

    ab = o3d.io.read_triangle_mesh(f"new_particles_63.obj")
    ab.rotate(rot_mat_compute.from_euler("y", 90, degrees=True).as_matrix())
    vis.add_geometry(ab)

    if "1" in comp:
        rot1 = comp["1"][0]["ab rot"]
        if rot1 > 90:
            rot_mat2 = rot_mat_compute.from_euler('x', -rot1+90,
                                                  degrees=True).as_matrix()
        else:
            rot_mat2 = rot_mat_compute.from_euler('x', rot1-90,
                                                  degrees=True).as_matrix()
        trans = comp["1"][0]["ab center"] - comp["1"][0]["head center"]
        ab.translate([0, -trans[1], -trans[0]])

        ab.rotate(rot_mat2, ab.get_center())
    if "2" in comp:
        rot2 = comp["2"][0]["ab rot"]
        rot_mat = rot_mat_compute.from_euler('z', rot2-90,
                                             degrees=True).as_matrix()
        ab.rotate(rot_mat, ab.get_center())
        trans = comp["2"][0]["ab center"] - comp["2"][0]["head center"]
        ab.translate([trans[1], trans[0], 0])

    vis.update_geometry(pcd)
    vis.update_geometry(ab)
    ab.scale(1000, ab.get_center())
    ab.translate([0, 0, 0])
    vis.get_view_control().set_zoom(2.5)
    vis.get_view_control().rotate(-500, 0)
    vis.capture_screen_image("v1.png", do_render=True)
    vis.get_view_control().rotate(500, 0)
    vis.get_view_control().set_zoom(2.5)

    vis.capture_screen_image("v2.png", do_render=True)
    vis.destroy_window()
    im1 = cv2.imread("v1.png")
    im2 = cv2.imread("v2.png")
    return im1, im2


def process(mypath='all_video'):
    """
    extract frames from videos stored in ../data_heavy/run
    """
    show_img = True
    all_folders = [f for f in listdir(mypath)]

    # check noise
    pixels = []
    for count, folder in enumerate(all_folders):
        all_files = [join(join(mypath, folder), f) for f in listdir(join(mypath, folder))]
        for video in all_files:
            name = "%s-%s.png" % (folder, video.split('/')[-1])
            json_name = "%s-%s.json" % (folder, video.split('/')[-1])
            img = cv2.imread(f"images/{name}")
            computation = read_json(f"images/{json_name}", img)
            view = name.split(".mp4.png")[0].split("-")[-1]
            if len(computation) == 0:
                continue
            if view == "1":
                pixels.append(computation["head pixels"])
    u = kmeans1d.cluster(pixels, 2)
    head_pixels_lower_bound = u.centroids[0]

    for count, folder in enumerate(all_folders):
        # if count != 3:
        #     continue
        all_files = [join(join(mypath, folder), f) for f in listdir(join(mypath, folder))]
        res = {}
        for video in all_files:
            name = "%s-%s.png" % (folder, video.split('/')[-1])
            json_name = "%s-%s.json" % (folder, video.split('/')[-1])
            img = cv2.imread(f"images/{name}")
            computation = read_json(f"images/{json_name}", img)
            view = name.split(".mp4.png")[0].split("-")[-1]
            if len(computation) == 0:
                continue
            if view == "1":
                if computation["head center"] is None:
                    computation["head pixels"] = computation["ab pixels"]
                    computation["head center"] = computation["ab center"]
                    computation["head rot"] = computation["ab rot"]
                if computation["head pixels"] <= head_pixels_lower_bound:
                    computation["head pixels"] = computation["ab pixels"]
                    computation["head center"] = computation["ab center"]

            res[view] = [computation, img]
        print(count, folder)
        if len([u[0] for u in res.values()]) == 0:
            continue
        # check if airbag exists
        if sum([u[0]["ab pixels"] for u in res.values()]) == 0:
            airbag = False
        else:
            airbag = True

        if not airbag:
            img_ori = np.hstack([res["1"][1], res["2"][1]])
            im1, im2 = visualize_wo_ab(res["1"][0], res["2"][0])
            im3 = np.hstack([im1, im2])
            im3 = cv2.resize(im3, (im3.shape[1]//2, im3.shape[0]//2))
            if show_img:
                cv2.imshow("", np.vstack([img_ori, im3]))
                cv2.waitKey()
                cv2.destroyAllWindows()
        else:
            img_ori = np.hstack([res["1"][1], res["2"][1]])
            im1, im2 = visualize(res["1"][0], res["2"][0])
            im3 = np.hstack([im1, im2])
            im3 = cv2.resize(im3, (im3.shape[1] // 2, im3.shape[0] // 2))
            if show_img:
                cv2.imshow("", np.vstack([img_ori, im3]))
                cv2.waitKey()
                cv2.destroyAllWindows()

        cv2.imwrite(f"all_video/{folder}/res.png", np.vstack([img_ori, im3]))


if __name__ == '__main__':
    process()
    # extract_all_video()
    # k_means_smoothing(None)
