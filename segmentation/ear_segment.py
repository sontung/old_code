import cv2
import numpy as np
import pickle
import os
from tqdm import tqdm
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from os import listdir
from os.path import isfile, join


class EAR:
    def __init__(self, checkpoint_dir="../data_heavy/ear.pth", threshold=0.5):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

        cfg.MODEL.WEIGHTS = checkpoint_dir
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        self.predictor = DefaultPredictor(cfg)

    def predict(self, img):
        predict = self.predictor(img)

        final_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        if predict["instances"].to("cpu").has("pred_masks"):
            masks = np.asarray(predict["instances"].to("cpu").pred_masks, dtype=np.uint8)

            if len(masks) >= 1:
                final_mask = masks[0]
                for mask in masks[1:]:
                    final_mask = cv2.bitwise_or(final_mask, mask)
        final_mask *= 255

        mask_img = img.copy()
        mask_img[final_mask == 0] *= 0
        # combine_img = np.hstack((img, mask_img, cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)))

        return final_mask, mask_img


if __name__ == "__main__":
    mypath = "../data_heavy/frames"
    frames = [f for f in listdir(mypath) if isfile(join(mypath, f)) if ".txt" not in f]
    saved_dir = "../data_heavy/frames_ear_only"
    saved_dir2 = "../data_heavy/frames_ear_coord_only"
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(saved_dir2, exist_ok=True)

    model = EAR()
    for im_name in tqdm(frames, desc="Extracting mask segment"):
        image = cv2.imread(join(mypath, im_name))
        mask_pre, combine_image = model.predict(image)
        cv2.imwrite(join(saved_dir, im_name), combine_image)

        # cv2.imshow("test", combine_image)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # pixels = []
        # for i in range(image.shape[0]):
        #     for j in range(image.shape[1]):
        #         if mask_pre[i, j] > 0:
        #             pixels.append((i, j))

        pixels = np.argwhere(mask_pre > 0)
        with open(join(saved_dir2, im_name), "wb") as fp:
            pickle.dump(pixels, fp)

