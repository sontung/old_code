import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.utils.visualizer import ColorMode


def main():
    register_coco_instances("airbag_train", {},
                            "/media/sontung/580ECE740ECE4B28/airbag/annotations/instances_train2017.json",
                            "/media/sontung/580ECE740ECE4B28/airbag/train2017")
    register_coco_instances("airbag_val", {},
                            "/media/sontung/580ECE740ECE4B28/airbag/annotations/instances_val2017.json",
                            "/media/sontung/580ECE740ECE4B28/airbag/val2017")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("airbag_train",)
    cfg.DATASETS.TEST = ("airbag_val", "airbag_train")
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    trainer = DefaultTrainer(cfg)


    # evaluation
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    flush_outputs(predictor)


# VALIDATION DATASET
def flush_outputs(predictor):
    print("flushing output images")

    dataset_dicts = load_coco_json("/media/sontung/580ECE740ECE4B28/airbag/annotations/instances_val2017.json",
                                   "/media/sontung/580ECE740ECE4B28/airbag/val2017",
                                   "airbag_val")
    balloon_metadata = MetadataCatalog.get("airbag_val")
    os.makedirs("output/airbag_val/", exist_ok=True)
    os.makedirs("output/airbag_train/", exist_ok=True)

    for d in random.sample(dataset_dicts, 200):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=balloon_metadata,
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        visualizer = Visualizer(im[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
        out2 = visualizer.draw_dataset_dict(d)
        res = np.vstack([out2.get_image()[:, :, ::-1], out.get_image()[:, :, ::-1]])

        cv2.imwrite("output/airbag_val/%s" % d["file_name"].split("/")[-1], res)

    # TRAIN DATASET
    dataset_dicts = load_coco_json("/media/sontung/580ECE740ECE4B28/airbag/annotations/instances_train2017.json",
                                   "/media/sontung/580ECE740ECE4B28/airbag/train2017",
                                   "airbag_train")
    balloon_metadata = MetadataCatalog.get("airbag_train")

    for d in random.sample(dataset_dicts, 200):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=balloon_metadata,
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        visualizer = Visualizer(im[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
        out2 = visualizer.draw_dataset_dict(d)
        res = np.vstack([out2.get_image()[:, :, ::-1], out.get_image()[:, :, ::-1]])

        cv2.imwrite("output/airbag_train/%s" % d["file_name"].split("/")[-1], res)


if __name__ == '__main__':
    main()