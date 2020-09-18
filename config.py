
import os
from utils import get_classes
from data import get_dataset
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam

input_shape = (416, 416)
scales = [32, 16, 8]
output_shapes = [(input_shape[0]//scale, input_shape[1]//scale) for scale in scales]

cwd = os.getcwd()
class_path = os.path.join(cwd, 'Dataset/classes.txt')
class_names = get_classes(class_path)
num_classes = len(class_names)

optimizer_learning_rate = 1e-4

dataset_path = os.path.join(cwd, 'Dataset')
image_extension = '.png'
dataset = get_dataset(dataset_path, num_classes, image_extension)

batch_size = 16

val_split = 0.1
num_val = int(len(dataset)*val_split)
num_train = len(dataset) - num_val

log_dir = os.path.join(cwd, 'Logs')
weights_path = os.path.join(log_dir, 'trained_weights_final.h5') 
logging = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5', 
                             monitor='val_loss', save_weights_only=True, save_best_only=True, save_freq='epoch', period=10)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

optimizer = Adam(learning_rate=optimizer_learning_rate)

nms_thresh = 0.3