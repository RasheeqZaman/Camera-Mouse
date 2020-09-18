import numpy as np
from PIL import Image
from cv2 import cv2
import os

def get_classes(classes_path):
  '''loads the classes'''
  with open(classes_path) as f:
    class_names = f.readlines()
  class_names = [c.strip() for c in class_names]
  return class_names


def smooth_one_hot(class_id, num_classes, delta=0.99):
  one_hot = np.zeros(num_classes)
  one_hot[class_id] = delta
  smooth = np.empty([num_classes]) 
  smooth.fill((1-delta)/num_classes)
  return np.array([one_hot, smooth]).sum(axis=0)


def preprocess_true_boxes(true_boxes, input_shape, output_shapes, num_classes):
  assert (true_boxes[..., 2]<num_classes).all(), 'class id must be less than num_classes'
  
  scales = len(output_shapes)
  input_shape = np.array(input_shape, dtype='int32')
  batch_size = true_boxes.shape[0]

  y_true = [np.zeros((batch_size, output_shapes[l][0], output_shapes[l][1], 3+num_classes),
        dtype='float32') for l in range(scales)]

  for b in range(batch_size):
    for l in range(scales):
      i = true_boxes[b,0]*output_shapes[l][0]
      j = true_boxes[b,1]*output_shapes[l][1]
      c = true_boxes[b,2].astype('int32')
      if c == -1: continue

      y_true[l][b, int(j), int(i), 0] = i - float(int(i))
      y_true[l][b, int(j), int(i), 1] = j - float(int(j))
      y_true[l][b, int(j), int(i), 2] = 1
      y_true[l][b, int(j), int(i), 3:] = smooth_one_hot(c, num_classes)

  return y_true


def preprocess_image(image, input_shape):
  return np.array(image.resize(input_shape))/255.


def get_data(data_line, input_shape, image_extension):
  image = Image.open(data_line[0]+image_extension)
  image_data = preprocess_image(image, input_shape)

  box_data = np.zeros((3))
  if data_line[1] == -1:
    box_data[2] = -1
    return image_data, box_data
  
  with open(data_line[0]+'.txt') as f:
    box_file = f.readline()
  box_file_data = box_file.split()
  box_data[0] = float(box_file_data[1])
  box_data[1] = float(box_file_data[2])
  box_data[2] = data_line[1]

  return image_data, box_data


class BoundBox:
    def __init__(self, x, y, objectness = None, classes = None):
        self.x = x
        self.y = y
        
        self.objectness = objectness
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score


def draw_boxes(image, box, label_names):
  if box is None: return

  label_index = box.get_label()
  label_str = str(label_index)+", "+"{:.2f}".format(box.objectness*100)+"%, "+"{:.2f}".format(box.get_score()*100)+"%"

  image = cv2.circle(image, (box.x, box.y), 2, (255, 0, 0), 2)
  image = cv2.putText(img=image, 
              text=label_str, 
              org=(box.x, box.y), 
              fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
              fontScale=0.3, 
              color=(0,0,0), 
              thickness=1)


def image_manupulate(input_path, output_path, desired_shape):
  label_paths = sorted([os.path.join(input_path, o) for o in os.listdir(input_path) if (os.path.isdir(os.path.join(input_path,o)))])

  for i in range(len(label_paths)):
    label_name = os.path.basename(os.path.normpath(label_paths[i]))
    label_path = os.path.join(output_path, label_name)
    for (root, _, filenames) in os.walk(label_paths[i]):
      for f in filenames:
        if f.endswith('.png'):
          image = Image.open(os.path.join(root, f))
          w, h = image.size
          scale = min(desired_shape[0]/w, desired_shape[1]/h)
          image = image.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
          dir_path = os.path.join(label_path, os.path.basename(os.path.normpath(root)))
          path = os.path.join(dir_path, f)
          image.save(path)