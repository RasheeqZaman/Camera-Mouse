import numpy as np
from PIL import Image
from cv2 import cv2

from yolov3 import get_yolo_boxes
from utils import draw_boxes, preprocess_image

def detect_from_image_batches(model, images, net_shape, nms_thresh, label_names, batch_output=None, video_batches=True):
    batch_boxes = get_yolo_boxes(model, images, net_shape, nms_thresh, batch_output)

    for i in range(len(images)):
        draw_boxes(images[i], batch_boxes[i], label_names)
        cv2.imshow('video with bboxes', images[i])
        if not video_batches: cv2.waitKey(0)
    
    if not video_batches: cv2.destroyAllWindows()

def detect_from_video(infer_model, net_shape, nms_thresh, label_names):
    video_reader = cv2.VideoCapture(0)
    video_reader.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_reader.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    test_batch_size = 1
    images = []

    
    while True:
        ret_val, image = video_reader.read()
        if ret_val == True: images.append(preprocess_image(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), net_shape))
        images = np.array(images)

        if (len(images)>=test_batch_size):
            detect_from_image_batches(infer_model, images, net_shape, nms_thresh, label_names)
            images = []
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()
