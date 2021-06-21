import sys
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from mypath import Path
from tqdm import trange
import os
from pycocotools.coco import COCO
from pycocotools import mask
from PIL import Image, ImageFile
from torchvision import transforms
from dataloaders import custom_transforms as tr
from dataloaders.utils import decode_segmap
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse
from modeling.deeplab import *
from dataloaders.utils import decode_seg_map_sequence
from utils.metrics import Evaluator
from tqdm import tqdm
import pandas as pd
from shutil import copyfile
import glob


ImageFile.LOAD_TRUNCATED_IMAGES = True

# sheet = pd.read_csv()


class COCOSegmentation(Dataset):
    NUM_CLASSES = 2
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
                1, 64, 20, 63, 7, 72]

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('coco'),
                 split='train',
                 year='2017'):
        super().__init__()
        ann_file = os.path.join(base_dir, 'annotations/instances_{}{}.json'.format(split, year))
        ids_file = os.path.join(base_dir, 'annotations/{}_ids_{}.pth'.format(split, year))
        self.img_dir = os.path.join(base_dir, 'images/{}{}'.format(split, year))
        self.split = split
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        if os.path.exists(ids_file):
            self.ids = torch.load(ids_file)
        else:
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)
        self.args = args

    def __getitem__(self, index):
        _img, _target, _name = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target, 'name': _name}

        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'val':
            sample2 = self.transform_val(sample)
            sample2['name'] = _name
            return sample2

    def _make_img_gt_point_pair(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        filename = os.path.basename(path)
        _img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        _target = Image.fromarray(self._gen_seg_mask(
            cocotarget, img_metadata['height'], img_metadata['width']))

        # _img = cv2.resize(np.array(_img), (513, 513))
        # _img = Image.fromarray(_img)
        # _target = cv2.resize(np.array(_target), (513, 513))
        # _target = Image.fromarray(_target)

        return _img, _target, filename

    def _preprocess(self, ids, ids_file):
        print("Preprocessing mask, this will take a while. " + \
              "But don't worry, it only run once for each split.")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'],
                                      img_metadata['width'])
            # more than 1k pixels
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'. \
                                 format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        torch.save(new_ids, ids_file)
        return new_ids

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            # tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __len__(self):
        return len(self.ids)


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


def inference():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    model = DeepLab(num_classes=21,
                    backbone="resnet",
                    output_stride=16,
                    sync_bn=None,
                    freeze_bn=True)
    checkpoint = torch.load("../../data_const/model_best.pth.tar")
    model.cuda().eval()
    model.load_state_dict(checkpoint['state_dict'])

    coco_val = COCOSegmentation(args, base_dir='/media/hblab/01D5F2DD5173DEA0/AirBag/airbag', split='val', year='2017')
    eval_loader = DataLoader(coco_val, batch_size=1, shuffle=True, num_workers=0)

    mIou = 0
    cnt = 0
    folder = 'origin'
    for ii, sample in enumerate(tqdm(eval_loader, desc="Evaluation images")):
        image, target, im_name = sample["image"], sample["label"], sample["name"]
        image = image.cuda()
        target = target.cuda()
        seg_tar = decode_seg_map_sequence(target.cpu().numpy(), dataset="coco") * 255
        with torch.no_grad():
            output = model(image)
            seg = decode_seg_map_sequence(torch.max(output, 1)[1].detach().cpu().numpy(), dataset="coco") * 255
            output = output.cpu()
            pred2 = torch.max(output, 1)[1]
            tem_pred = np.argmax(output.data.cpu().numpy(), axis=1)
            tem_target = target.cpu().numpy()

            for idx in range(seg.shape[0]):

                # calculate iou for each image
                pre = tem_pred[idx]
                tar = tem_target[idx]
                iou = iou_matrix(tar, pre)
                # print(f"{im_name[idx]} - {iou}", file=save_file)
                mIou += iou
                # if iou <= 0.5:
                #     folder = 'smaller_0.5'
                # else:
                #     folder = 'larger_0.5'
                pre_by_class = seg[idx].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                Image.fromarray(pre_by_class).save(f'/media/hblab/01D5F2DD5173DEA0/AirBag/segment/pre/{im_name[idx]}')
                # sys.exit()
                # tar_by_class = seg_tar[idx].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                # img = np.transpose(image[idx].cpu().numpy(), axes=[1, 2, 0])
                # img *= (0.229, 0.224, 0.225)
                # img += (0.485, 0.456, 0.406)
                # img *= 255.0
                # img = img.astype(np.uint8)
                #
                # merge_img = np.hstack([img, tar_by_class, pre_by_class])
                # new_name = f"{im_name[idx].replace('.png', '')}-{iou}.png"
                # Image.fromarray(merge_img).save(f"/media/hblab/01D5F2DD5173DEA0/AirBag/segment/{folder}/{new_name}")
                cnt += 1
                # sys.exit()
    print(mIou, cnt, mIou/cnt)
    return


def sort_by_iou():
    origin_fd = '/media/hblab/01D5F2DD5173DEA0/AirBag/segment/origin'
    dst_fd = '/media/hblab/01D5F2DD5173DEA0/AirBag/segment/sorted'
    os.makedirs(dst_fd, exist_ok=True)
    filename = os.listdir(origin_fd)
    filename = sorted(filename, key=lambda x: float(x.split('-')[-1].replace('.png', '')))

    for i in range(len(filename)):
        name = filename[i]
        new_name = f"{i}-{name.split('-')[-1]}"
        or_path = os.path.join(origin_fd, name)
        ne_path = os.path.join(dst_fd, new_name)

        copyfile(or_path, ne_path)

    return


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



if __name__ == "__main__":
    evaluate()
