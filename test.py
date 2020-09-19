
import numpy as np
from PIL import Image

import config as cfg
from yolov3 import Yolo
from detect import detect_from_video, detect_from_image_batches
from data import data_generator


"""# Model Initialize"""

yolo = Yolo(batch_size=cfg.batch_size)
model = yolo.create_model(cfg.input_shape, cfg.num_classes)


"""# Compile"""

model.compile(optimizer=cfg.optimizer, loss=yolo.calc_loss)
#print(model.summary())


"""# Load Weights"""

model.load_weights(cfg.weights_path)


"""# Detect"""

detect_from_video(model, cfg.input_shape, cfg.nms_thresh, cfg.class_names)
'''
val_data_generator = data_generator(cfg.dataset[cfg.num_train:cfg.num_train+100], cfg.input_shape, cfg.output_shapes, cfg.num_classes, cfg.image_extension)
detect_from_image_batches(model, val_data_generator[0], cfg.input_shape, cfg.nms_thresh, cfg.class_names, video_batches=False)
'''