from collections import namedtuple
from utils.metrics import Evaluator
from modeling.deeplab import *
from torch.utils.data import DataLoader, Dataset
from dataloaders import custom_transforms as tr
from torchvision import transforms
from dataloaders.utils import decode_seg_map_sequence
from PIL import Image
from dataloaders.datasets import coco
import glob
import numpy as np
import os
import cv2
import post_process
from tqdm import tqdm


class TestDataset(Dataset):
    def __init__(self, img_dir="../../data_heavy/frames"):
        super(TestDataset, self).__init__()
        self.img_dir = img_dir
        self.all_path_files = glob.glob(self.img_dir + '/*.png')
        print("loaded %s images" % len(self.all_path_files))

    def __len__(self):
        return len(self.all_path_files)

    def __getitem__(self, ind):
        _img = np.array(Image.open(self.all_path_files[ind]).convert('RGB'))
        original_img = np.array(_img)
        _img = Image.fromarray(original_img)
        sample = {'image': _img, "label": _img, "name": self.all_path_files[ind]}
        sample2 = self.transform_val(sample)
        sample2["name"] = self.all_path_files[ind]
        sample2["ori_img"] = original_img
        return sample2 

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)


def merge_2images(_im1, _im2, c1, c2):
    res = np.zeros(_im1.shape, dtype=np.uint8)
    indices = np.argwhere(_im1[:, :] == c1)
    res[indices[:, 0], indices[:, 1]] = c1
    indices = np.argwhere(_im2[:, :] == c2)
    res[indices[:, 0], indices[:, 1]] = c2
    return res


def check_contour(_image, _color):
    _new_image = np.zeros(_image.shape, dtype=np.uint8)

    gray_img = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
    cnts, _ = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    angle = None
    center = None
    x1, y1, x2, y2 = 0, 0, 0, 0
    if len(cnts) > 0:
        cv2.drawContours(_image, cnts, -1, (255, 0, 0), 3)
        largest_cnt = max(cnts, key=lambda du1: cv2.contourArea(du1))
        cv2.fillPoly(_new_image, pts=[largest_cnt], color=(192, 128, 128))
        ori_rect = cv2.minAreaRect(largest_cnt)
        center = (int(ori_rect[0][0]), int(ori_rect[0][1]))

        oriented_rect = cv2.boxPoints(ori_rect)
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
        # cv2.drawContours(_image, [oriented_rect], 0,  (255, 0, 0), 2)
        # cv2.circle(_image, (x1, y1), 3, (255, 255, 255), -1)
        # cv2.circle(_image, (x2, y2), 3, (255, 255, 255), -1)
        #
        # cv2.imshow('contour', np.hstack([_image, _new_image]))
        # cv2.waitKey()
        # cv2.destroyAllWindows()

    return _new_image, center, angle, x1, y1, x2, y2


def main(has_labels=False):
    args_tuple = namedtuple("Args", ["base_size", "crop_size"])
    model = DeepLab(num_classes=21,
                            backbone="resnet",
                            output_stride=16,
                            sync_bn=None,
                            freeze_bn=True)
    checkpoint = torch.load("../../data_const/model_best.pth.tar")
    model.cuda().eval()
    model.load_state_dict(checkpoint['state_dict'])
    args = args_tuple(513, 513)

    test_set = TestDataset()
    if has_labels:
        val_set = coco.COCOSegmentation(args, split="val")
        test_loader = DataLoader(val_set, batch_size=8, shuffle=False)
    else:
        test_loader = DataLoader(test_set, batch_size=8, shuffle=False)
    evaluator = Evaluator(21)
    os.makedirs("../../data_heavy/frames_seg_abh/", exist_ok=True)
    os.makedirs("../../data_heavy/frames_seg_abh_vis/", exist_ok=True)

    classid2color = np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                           [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                           [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                           [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                           [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                           [0, 64, 128]])
    count = 0
    with open("../../data_heavy/head_masks.txt", "w") as fp2:
        with open("../../data_heavy/frame2ab.txt", "w") as fp:
            for bid, sample in enumerate(tqdm(test_loader, desc="Extracting semantic masks")):

                if has_labels:
                    image, target, ori_img = sample['image'], sample['label'], sample["ori_img"]
                    image = image.cuda()
                    target = target.cuda()
                    with torch.no_grad():
                        output = model(image)
                    pred = output.data.cpu().numpy()
                    target = target.cpu().numpy()
                    pred = np.argmax(pred, axis=1)
                    for ind in range(pred.shape[0]):
                        count+=1
                        post_process.post_process(pred[ind],
                                                  sample["ori_img"][ind].numpy(),
                                                  count)
                    evaluator.add_batch(target, pred)
                    continue

                image, target, im_name = sample['image'], sample['label'], sample["name"]
                original_frames = sample["ori_img"]
                image = image.cuda()
                with torch.no_grad():
                    output = model(image)
                    seg = decode_seg_map_sequence(torch.max(output, 1)[1].detach().cpu().numpy(), dataset="coco")*255
                    output = output.cpu()
                    pred2 = torch.max(output, 1)[1]
                    for idx in range(seg.shape[0]):
                        img = seg[idx].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

                        new_img_ab = np.zeros(img.shape, dtype=np.uint8)
                        new_img_head = np.zeros(img.shape, dtype=np.uint8)

                        new_img_head[(img == classid2color[14]).all(axis=2)] = classid2color[14]
                        new_img_ab[(img == classid2color[15]).all(axis=2)] = classid2color[15]
                        im1, center1, rot1, x1, y1, x2, y2 = check_contour(new_img_head, classid2color[14].astype(np.uint8))
                        im2, center2, rot2, _, _, _, _ = check_contour(new_img_ab, classid2color[15].astype(np.uint8))
                        im_final = merge_2images(im1, im2,
                                                 classid2color[14].astype(np.uint8), classid2color[15].astype(np.uint8))

                        dist_x = -1
                        dist_y = -1
                        if center2 is not None and center1 is not None:
                            dist_x = center2[0]
                            dist_y = center2[1]

                        imn = im_name[idx].split("/")[-1]
                        ab_pixels = np.sum((pred2[idx]==15).numpy())
                        head_pixels = np.sum((pred2[idx]==14).numpy())
                        print(imn, ab_pixels, head_pixels, dist_x, dist_y, rot1, rot2, file=fp)
                        print(imn, x1, y1, x2, y2, file=fp2)

                        Image.fromarray(im_final).save("../../data_heavy/frames_seg_abh/%s" % imn)
                        original_frame = original_frames[idx].cpu().numpy().astype(np.uint8)
                        blend2 = cv2.addWeighted(original_frame, 0.3, im_final, 0.7, 0)
                        Image.fromarray(blend2).save("../../data_heavy/frames_seg_abh_vis/%s" % imn)

    # Fast test during the training
    if has_labels:
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        print(mIoU, FWIoU)


if __name__ == "__main__":
    main()
