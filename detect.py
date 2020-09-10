import numpy as np
from PIL import Image
from cv2 import cv2

from yolov3 import get_yolo_boxes
from utils import draw_boxes

def detect_from_image_batches(model, images, net_shape, nms_thresh, label_names):
    batch_boxes = get_yolo_boxes(model, images, net_shape, nms_thresh)

    for i in range(len(images)):
        draw_boxes(images[i], batch_boxes[i], label_names)
        cv2.imshow('video with bboxes', np.float32(images[i]))

def detect_from_video(infer_model, net_shape, nms_thresh, label_names):
    video_reader = cv2.VideoCapture(0)

    test_batch_size = 1
    images = []
    while True:
        ret_val, image = video_reader.read()
        if ret_val == True: images += [Image.fromarray(np.uint8(image))]

        if (len(images)>=test_batch_size):
            detect_from_image_batches(infer_model, images, net_shape, nms_thresh, label_names)
            images = []
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()