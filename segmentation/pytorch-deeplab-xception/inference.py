from collections import namedtuple
from utils.metrics import Evaluator
import torch
from modeling.deeplab import *
from torch.utils.data import DataLoader, Dataset
from dataloaders.datasets import cityscapes, coco
from dataloaders import custom_transforms as tr
from torchvision.utils import make_grid
from torchvision import transforms
from dataloaders.utils import decode_seg_map_sequence
from PIL import Image
from dataloaders.datasets import coco
import glob
import numpy as np
import os
import cv2
import sys
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
        _img = cv2.resize(np.array(_img), (513, 513))
        _img = Image.fromarray(_img)
        sample = {'image': _img, "label": _img, "name": self.all_path_files[ind]}
        sample2 = self.transform_val(sample)
        sample2["name"] = self.all_path_files[ind]
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
    if len(cnts) > 0:
        cv2.drawContours(_image, cnts, -1, (255, 0, 0), 3)
        largest_cnt = max(cnts, key=lambda du1: cv2.contourArea(du1))
        cv2.fillPoly(_new_image, pts=[largest_cnt], color=(192, 128, 128))
        ori_rect = cv2.minAreaRect(largest_cnt)
        center = (int(ori_rect[0][0]), int(ori_rect[0][1]))
        width = int(ori_rect[1][0])
        height = int(ori_rect[1][1])
        angle = int(ori_rect[2])

        if width < height:
            angle = 90 - angle
        else:
            angle = - angle

        oriented_rect = cv2.boxPoints(ori_rect)
        oriented_rect = np.int0(oriented_rect)
        # cv2.drawContours(_new_image, [oriented_rect], 0,  (255, 0, 0), 2)
        # cv2.imshow('contour', np.hstack([_image, _new_image]))
        # cv2.waitKey()
        # cv2.destroyAllWindows()

    return _new_image, center, angle


def main():
    Args = namedtuple("Args", ["base_size", "crop_size"])
    model = DeepLab(num_classes=21,
                            backbone="resnet",
                            output_stride=16,
                            sync_bn=None,
                            freeze_bn=True)
    checkpoint = torch.load("../../data_const/model_best.pth.tar")
    model.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    HAS_LABELS = False
    args = Args(513, 513)

    test_set = TestDataset()
    if HAS_LABELS:
        val_set = coco.COCOSegmentation(args, split="val")
        test_loader = DataLoader(val_set, batch_size=8, shuffle=False)
    else:
        test_loader = DataLoader(test_set, batch_size=8, shuffle=False)
    evaluator = Evaluator(21)
    os.makedirs("../../data_heavy/frames_seg_abh/", exist_ok=True)
    classid2color = np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                           [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                           [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                           [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                           [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                           [0, 64, 128]])
    with open("../../data_heavy/frame2ab.txt", "w") as fp:
        for bid, sample in enumerate(tqdm(test_loader, desc="Extracting semantic masks")):
            image, target, im_name = sample['image'], sample['label'], sample["name"]
            image = image.cuda()
            target = target.cuda()
            with torch.no_grad():
                output = model(image)
                seg = decode_seg_map_sequence(torch.max(output, 1)[1].detach().cpu().numpy(), dataset="coco")*255
                output = output.cpu()
                pred2 = torch.max(output, 1)[1]
                for idx in range(seg.shape[0]):
                    # print(np.unique(pred2[idx]))
                    img = seg[idx].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

                    new_img_ab = np.zeros(img.shape, dtype=np.uint8)
                    new_img_head = np.zeros(img.shape, dtype=np.uint8)

                    new_img_head[(img == classid2color[14]).all(axis=2)] = classid2color[14]
                    new_img_ab[(img == classid2color[15]).all(axis=2)] = classid2color[15]

                    # cv2.imshow("test", np.hstack([img, new_img_ab, new_img_head]))
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()

                    im1, center1, rot1 = check_contour(new_img_head, classid2color[14].astype(np.uint8))
                    im2, center2, rot2 = check_contour(new_img_ab, classid2color[15].astype(np.uint8))
                    im_final = merge_2images(im1, im2,
                                             classid2color[14].astype(np.uint8), classid2color[15].astype(np.uint8))
                    dist = -1
                    if center2 is not None and center1 is not None:
                        dist = center2[0]-center1[0]

                    imn = im_name[idx].split("/")[-1]
                    ab_pixels = np.sum((pred2[idx]==15).numpy())
                    print(imn, ab_pixels, dist, rot1, rot2, file=fp)
                    Image.fromarray(im_final).save("../../data_heavy/frames_seg_abh/%s" % imn)


            #segall = make_grid(seg, 8, normalize=False, range=(0, 255))
            #segall = segall.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            #imall = make_grid(image, 8, normalize=False, range=(0, 255))
            #imall = imall.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            #Image.fromarray(np.hstack([segall, imall])).save("outputs/s%d.png" % bid)
            if HAS_LABELS:
                pred = output.data.cpu().numpy()
                target = target.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                evaluator.add_batch(target, pred)

    # Fast test during the training
    if HAS_LABELS:
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        print(mIoU, FWIoU)


if __name__ == "__main__":
    main()
