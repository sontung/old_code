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
    def __init__(self, checkpoint_dir="../data_const/ear.pth", threshold=0.5):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

        cfg.MODEL.WEIGHTS = checkpoint_dir
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        self.predictor = DefaultPredictor(cfg)

    def predict(self, img, im_name):
        predict = self.predictor(img)

        final_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        if predict["instances"].to("cpu").has("pred_masks"):
            masks = np.asarray(predict["instances"].to("cpu").pred_masks, dtype=np.uint8)

            if len(masks) >= 1:
                final_mask = masks[0]
                for mask in masks[1:]:
                    final_mask = cv2.bitwise_or(final_mask, mask)
        final_mask *= 255
        self.post_process(final_mask, im_name)

        mask_img = img.copy()
        mask_img[final_mask == 0] *= 0

        return final_mask, mask_img

    def inside_head(self, cnt, head):
        x0, y0 = np.mean(cnt[:, :, 0]), np.mean(cnt[:, :, 1])
        print("cnt", x0, y0)
        print(cnt)
        if np.min(head[0]) <= x0 <= np.max(head[1]):
            return True
        else:
            return False

    def post_process(self, mask, im_name):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        res = np.zeros_like(mask)
        seg_dir = "../data_heavy/frames_seg_abh"

        if len(contours) > 0:
            cv2.drawContours(res, contours, -1, (255, 255, 0), 3)

            # filter head
            seg_im = cv2.imread(f"{seg_dir}/{im_name}")
            seg_im = cv2.cvtColor(seg_im, cv2.COLOR_BGR2RGB)

            arr = seg_im == [64, 128, 128]
            print(np.nonzero(arr))
            print(contours[0].shape)

            for cnt in contours:
                if self.inside_head(cnt, np.nonzero(arr)):
                    print("draw")
                    cv2.drawContours(mask, [cnt], 0, (0, 255, 0), 3)
                    cv2.imshow("test", mask)
                    cv2.waitKey()
                    cv2.destroyAllWindows()


if __name__ == "__main__":
    mypath = "../data_heavy/frames"
    frames = [f for f in listdir(mypath) if isfile(join(mypath, f)) if ".txt" not in f]
    saved_dir = "../data_heavy/frames_ear_only"
    saved_dir2 = "../data_heavy/frames_ear_coord_only"
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(saved_dir2, exist_ok=True)

    model = EAR()
    for im_name in tqdm(frames, desc="Extracting ear mask segment"):
        image = cv2.imread(join(mypath, im_name))
        mask_pre, combine_image = model.predict(image, im_name)
        cv2.imwrite(join(saved_dir, im_name), combine_image)

        print(mask_pre.shape, np.unique(mask_pre))

        # cv2.imshow("test", combine_image)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        pixels = np.argwhere(mask_pre > 0)
        with open(join(saved_dir2, im_name), "wb") as fp:
            pickle.dump(pixels, fp)

        break
