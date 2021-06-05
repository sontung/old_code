
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
import glob
import numpy as np


class TestDataset(Dataset):
    def __init__(self, img_dir="/home/sontung/work/3d-air-bag-p2/data_heavy/frames"):
        super(TestDataset, self).__init__()
        self.img_dir = img_dir
        self.all_path_files = glob.glob(self.img_dir + '/*.png')
        print("loaded %s images" % len(self.all_path_files))

    def __len__(self):
        return len(self.all_path_files)

    def __getitem__(self, ind):
        _img = Image.open(self.all_path_files[ind]).convert('RGB')
        sample = {'image': _img, "label": _img, "name": self.all_path_files[ind]}
        return self.transform_val(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=513),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)


model = DeepLab(num_classes=21,
                        backbone="resnet",
                        output_stride=16,
                        sync_bn=None,
                        freeze_bn=True)
checkpoint = torch.load("run/coco/deeplab-resnet/model_best.pth.tar")
model.cuda()
model.load_state_dict(checkpoint['state_dict'])

test_set = TestDataset()
test_loader = DataLoader(test_set, batch_size=8, shuffle=False)
evaluator = Evaluator(21)
for bid, sample in enumerate(test_loader):
    image, target = sample['image'], sample['label']
    image = image.cuda()
    with torch.no_grad():
        output = model(image)
        seg = decode_seg_map_sequence(torch.max(output, 1)[1].detach().cpu().numpy(), dataset="coco")
        Image.fromarray(seg[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)).save("outputs/%d.png" % bid)
        segall = make_grid(seg, 8, normalize=False, range=(0, 255))
        segall = segall.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        imall = make_grid(image, 8, normalize=False, range=(0, 255))
        imall = imall.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        Image.fromarray(np.hstack([segall, imall])).save("outputs/s%d.png" % bid)
        
        pred = output.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        evaluator.add_batch(target, pred)
                                                                                              
# Fast test during the training
Acc = evaluator.Pixel_Accuracy()
Acc_class = evaluator.Pixel_Accuracy_Class()
mIoU = evaluator.Mean_Intersection_over_Union()
FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
