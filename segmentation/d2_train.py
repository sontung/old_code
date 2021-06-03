import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances, load_coco_json


register_coco_instances("airbag_train", {},
                        "/media/sontung/580ECE740ECE4B28/airbag/annotations/instances_train2017.json",
                        "/media/sontung/580ECE740ECE4B28/airbag/train2017")
register_coco_instances("airbag_val", {},
                        "/media/sontung/580ECE740ECE4B28/airbag/annotations/instances_val2017.json",
                        "/media/sontung/580ECE740ECE4B28/airbag/val2017")

dataset_dicts = load_coco_json(
    "/media/sontung/580ECE740ECE4B28/airbag/annotations/instances_train2017.json",
    "/media/sontung/580ECE740ECE4B28/airbag/train2017",
    "airbag_train")

balloon_metadata = MetadataCatalog.get("airbag_train")
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow("test", out.get_image()[:, :, ::-1])
    cv2.waitKey()
    cv2.destroyAllWindows()


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("airbag_train",)
cfg.DATASETS.TEST = ("airbag_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 1000     # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []         # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
