import torch
from modeling.deeplab import *
from torch.utils.data import DataLoader, Dataset
from dataloaders.datasets import cityscapes, coco
from dataloaders import custom_transforms as tr
from torchvision import transforms
from PIL import Image
import glob


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
        sample = {'image': _img, "label": _img}
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

for sample in test_loader:
    image, target = sample['image'], sample['label']
    image = image.cuda()
    with torch.no_grad():
       output = model(image)



if i % (num_img_tr // 10) == 0:
    global_step = i + num_img_tr * epoch
    self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)
