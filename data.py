
import os
import random
import config as cfg
from tqdm import tqdm
import numpy as np

from utils import get_data, preprocess_true_boxes

def get_dataset(path, num_classes, image_extension):
    label_paths = sorted([os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path,o)) and o != '.ipynb_checkpoints'])
    if num_classes+1 != len(label_paths): return None

    dataset = []
    for i in range(num_classes+1):
        for (root, _, filenames) in os.walk(label_paths[i]):
            for j in range(len(filenames)):
                if filenames[j].endswith(image_extension):
                    dataset.append((os.path.splitext(os.path.join(root, filenames[j]))[0], i-1))

    random.seed(10101)
    random.shuffle(dataset)
    return dataset

def data_generator(dataset, input_shape, output_shape, num_classes, image_extenstion):
    n = len(dataset)
    if n==0 or cfg.batch_size<=0: return None

    image_data = []
    box_data = []
    for i in tqdm(range(n)):
        image, box = get_data(dataset[i], input_shape, image_extenstion)
        image_data.append(image)
        box_data.append(box)

    image_data = np.array(image_data)
    box_data = np.array(box_data)
    y_true = preprocess_true_boxes(box_data, input_shape, output_shape, num_classes)

    return image_data, y_true