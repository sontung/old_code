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

class TestDataset(Dataset):
    def __init__(self, img_dir="/home/sontung/work/3d-air-bag-p2/data_heavy/frames"):
        super(TestDataset, self).__init__()
        self.img_dir = img_dir
        self.all_path_files = glob.glob(self.img_dir + '/*.png')
        print("loaded %s images" % len(self.all_path_files))

    def __len__(self):
        return len(self.all_path_files)

    def __getitem__(self, ind):
        _img = np.array(Image.open(self.all_path_files[ind]).convert('RGB'))
        #_img = cv2.resize(np.array(_img), (_img.shape[0]//4, _img.shape[1]//4))
        _img = cv2.resize(np.array(_img), (513, 513))
        _img = Image.fromarray(_img)
        sample = {'image': _img, "label": _img, "name": self.all_path_files[ind]}
        sample2 = self.transform_val(sample)
        sample2["name"] = self.all_path_files[ind]
        return sample2 

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
#            tr.FixScaleCrop(crop_size=513),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

Args = namedtuple("Args", ["base_size", "crop_size"])
model = DeepLab(num_classes=21,
                        backbone="resnet",
                        output_stride=16,
                        sync_bn=None,
                        freeze_bn=True)
checkpoint = torch.load("run/coco/deeplab-resnet/model_best.pth.tar")
model.cuda()
model.load_state_dict(checkpoint['state_dict'])
HAS_LABELS = False
args = Args(513, 513)

val_set = coco.COCOSegmentation(args, split="val")
test_set = TestDataset()
if HAS_LABELS:
    test_loader = DataLoader(val_set, batch_size=8, shuffle=False)
else:
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False)
evaluator = Evaluator(21)
os.makedirs("../../data_heavy/frames_seg_abh/", exist_ok=True)
for bid, sample in enumerate(test_loader):
    image, target, im_name = sample['image'], sample['label'], sample["name"]
    image = image.cuda()
    target = target.cuda()
    with torch.no_grad():
        output = model(image)
        seg = decode_seg_map_sequence(torch.max(output, 1)[1].detach().cpu().numpy(), dataset="coco")*255
        for idx in range(seg.shape[0]):
            imn = im_name[idx].split("/")[-1]
            Image.fromarray(seg[idx].permute(1, 2, 0).cpu().numpy().astype(np.uint8)).save("../../data_heavy/frames_seg_abh/%s" % imn)
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
