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

        seg_im = cv2.imread(f"../data_heavy/frames_seg_abh/{im_name}")
        head_only_im = np.zeros(seg_im.shape, dtype=np.uint8)
        head_indices = (seg_im == (128, 128, 64)).all(axis=2).astype(np.uint8)
        final_mask_filtered = self.inside_head(final_mask, np.nonzero(head_indices))
        head_only_im[(seg_im == (128, 128, 64)).all(axis=2)] = 255

        # cv2.imshow("t", np.hstack([final_mask_filtered, final_mask]))
        # cv2.imshow("t2", np.hstack([img, head_only_im]))
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        mask_img = img.copy()
        mask_img[final_mask_filtered == 0] *= 0

        return final_mask_filtered, mask_img

    def inside_head(self, ear_mask, head):
        res = np.zeros_like(ear_mask, np.uint8)
        head_x, head_y = head
        if len(head_x) == 0:
            return res
        contours, _ = cv2.findContours(ear_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x0, y0 = np.mean(cnt[:, :, 0]), np.mean(cnt[:, :, 1])
            if np.min(head_x) <= y0 <= np.max(head_x) and np.min(head_y) <= x0 <= np.max(head_y):
                cv2.fillPoly(res, pts=[cnt], color=[255])
                return res
        return res

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
    import glob

    mypath = "../data_heavy/frames"
    # frames = [f for f in listdir(mypath) if isfile(join(mypath, f)) if ".txt" not in f]
    frames = glob.glob("../data_heavy/frames/1-*.png")
    saved_dir = "../data_heavy/frames_ear_only"
    saved_dir2 = "../data_heavy/frames_ear_coord_only"
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(saved_dir2, exist_ok=True)

    model = EAR()
    for im_name in tqdm(frames, desc="Extracting ear mask segment"):
        im_name = im_name.split("/")[-1]
        image = cv2.imread(join(mypath, im_name))
        mask_pre, combine_image = model.predict(image, im_name)
        cv2.imwrite(join(saved_dir, im_name), combine_image)

        # cv2.imshow("test", combine_image)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        pixels = np.argwhere(mask_pre > 0)
        with open(join(saved_dir2, im_name), "wb") as fp:
            pickle.dump(pixels, fp)