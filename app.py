from flask import Flask, render_template, request
from waitress import serve
import re
import numpy as np

import detectron2
import os, cv2
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import torch

app = Flask(__name__)
app.secret_key = 'app_secret'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://postgres:root@localhost:5433/postgres"

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("pod_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300 #3000   # 300 iterations se0ems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (pod).

cfg.MODEL.WEIGHTS = os.path.join(r"./model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
predictor = DefaultPredictor(cfg)

@app.route('/getPods', methods=['POST'])
def newsFeed():
    imagefile = request.files.get('imagefile', '')
    file_bytes = np.fromfile(imagefile, np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    outputs = predictor(image)

    seedsInSoy=0

    masks = outputs['instances'].pred_masks
    soySizes = (torch.sum(torch.flatten(masks, start_dim=1),dim=1))

    for seed in (soySizes.detach().cpu().numpy()):
        if seed < 50:
          seedsInSoy = seedsInSoy + 1
        if seed > 50 and seed < 100:
          seedsInSoy = seedsInSoy + 2
        if seed > 100 and seed < 150:
          seedsInSoy = seedsInSoy + 3
        if seed > 150:
          seedsInSoy = seedsInSoy + 4

    return {'foundPods': str(len(outputs.get("instances"))), "seedsInSoy": str(seedsInSoy)}


if __name__ == '__main__':
    serve(app, host='127.0.0.1',port=5000)