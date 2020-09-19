
import sys
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from keras.losses import MeanSquaredError, BinaryCrossentropy
from keras.layers import Input, Conv2D, ZeroPadding2D, Add, UpSampling2D, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from PIL import Image

from utils import BoundBox, preprocess_image

class Yolo():

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.obj_scale = 1
        self.noobj_scale = 100
        self.mse = MeanSquaredError()
        self.bce = BinaryCrossentropy()

    def calc_loss(self, y_true, y_pred):

        x = tf.math.sigmoid(y_pred[..., 0])
        y = tf.math.sigmoid(y_pred[..., 1])
        pred_conf = tf.math.sigmoid(y_pred[..., 2])
        pred_cls = tf.math.sigmoid(y_pred[..., 3:])
        
        tx = y_true[..., 0]
        ty = y_true[..., 1]
        tconf = y_true[..., 2]
        tcls = y_true[..., 3:]

        obj_mask = tf.cast(y_true[..., 2], tf.bool)
        noobj_mask = tf.cast(1-y_true[..., 2], tf.bool)

        loss_x = self.mse(tf.boolean_mask(tx, obj_mask), tf.boolean_mask(x, obj_mask))
        loss_y = self.mse(tf.boolean_mask(ty, obj_mask), tf.boolean_mask(y, obj_mask))
        loss_conf_obj = self.bce(tf.boolean_mask(tconf, obj_mask), tf.boolean_mask(pred_conf, obj_mask))
        loss_conf_noobj = self.bce(tf.boolean_mask(tconf, noobj_mask), tf.boolean_mask(pred_conf, noobj_mask))
        loss_conf = self.obj_scale*loss_conf_obj + self.noobj_scale*loss_conf_noobj
        loss_cls = self.bce(tf.boolean_mask(tcls, obj_mask), tf.boolean_mask(pred_cls, obj_mask))
        loss = (loss_x + loss_y + loss_conf + loss_cls)

        return loss

    def DarknetConv2D_BN_Leaky(self, x, num_filters, kernel_size, strides=(1,1), alpha=0.1, l2_penalty=5e-4):
        """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
        x = Conv2D(num_filters, kernel_size, strides=strides, padding='valid' if strides==(2,2) else 'same', use_bias=False, kernel_regularizer=l2(l2_penalty))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=alpha)(x)
        return x

    def resblock_body(self, x, num_filters, num_blocks):
        '''A series of resblocks starting with a downsampling Convolution2D'''
        # Darknet uses left and top padding instead of 'same' mode
        x = ZeroPadding2D(((1,0),(1,0)))(x)
        x = self.DarknetConv2D_BN_Leaky(x, num_filters, (3,3), strides=(2,2))
        x_shortcut = x
        for _ in range(num_blocks):
            x = self.DarknetConv2D_BN_Leaky(x_shortcut, num_filters//2, (1,1))
            x = self.DarknetConv2D_BN_Leaky(x, num_filters, (3,3))
            x_shortcut = Add()([x,x_shortcut])
        return x_shortcut

    def darknet_body(self, x):
        '''Darknent body having 52 Convolution2D layers'''
        x = self.DarknetConv2D_BN_Leaky(x, 32, (3,3))
        
        x = self.resblock_body(x, 64, 1)
        x = self.resblock_body(x, 128, 2)
        x = self.resblock_body(x, 256, 8)
        x = self.resblock_body(x, 512, 8)
        x = self.resblock_body(x, 1024, 4)
        
        return x

    def make_last_layers(self, x, num_filters, out_filters):
        '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
        x = self.DarknetConv2D_BN_Leaky(x, num_filters, (1,1))
        x = self.DarknetConv2D_BN_Leaky(x, num_filters*2, (3,3))
        x = self.DarknetConv2D_BN_Leaky(x, num_filters, (1,1))
        x = self.DarknetConv2D_BN_Leaky(x, num_filters*2, (3,3))
        x = self.DarknetConv2D_BN_Leaky(x, num_filters, (1,1))

        y = self.DarknetConv2D_BN_Leaky(x, num_filters*2, (3,3))
        y = Conv2D(out_filters, (1,1), padding='same', use_bias=True, kernel_regularizer=l2(5e-4))(y)
        return x, y

    def yolo_body(self, inputs, num_classes):
        """Create YOLO_V3 model CNN body in Keras."""
        darknet = Model(inputs, self.darknet_body(inputs))

        '''
        for i in range(len(darknet.layers)):
            print(i, darknet.layers[i].output.shape)
        '''

        x, y1 = self.make_last_layers(darknet.output, 512, num_classes+3)

        x = self.DarknetConv2D_BN_Leaky(x, 256, (1,1))
        x = UpSampling2D(2)(x)
        x = Concatenate()([x,darknet.layers[152].output])
        x, y2 = self.make_last_layers(x, 256, num_classes+3)

        x = self.DarknetConv2D_BN_Leaky(x, 128, (1,1))
        x = UpSampling2D(2)(x)
        x = Concatenate()([x,darknet.layers[92].output])
        x, y3 = self.make_last_layers(x, 128, num_classes+3)

        return Model(inputs, [y1,y2,y3])

    def create_model(self, input_shape, num_classes):
        '''create the training model'''
        K.clear_session() # get a new session
        w, h = input_shape
        image_input = Input(shape=(w, h, 3))

        model_body = self.yolo_body(image_input, num_classes)
        print('Create YOLOv3 model with {} classes.'.format(num_classes))
        return model_body



def decode_netout(batch_output, image_index, net_shape, nms_thresh):
    boxes = []
    for batch_netout in batch_output:
        netout = batch_netout[image_index]

        grid_h, grid_w = netout.shape[:2]

        netout[..., :]  = tf.math.sigmoid(netout[..., :])

        for row in range(grid_w):
            for col in range(grid_h):
                
                objectness = netout[row, col, 2]
                if(objectness <= nms_thresh): continue

                # first 2 elements are x, y
                x, y = netout[row,col,:2]

                x = (col + x) / grid_w # center position, unit: image width
                y = (row + y) / grid_h # center position, unit: image height

                # last elements are class probabilities
                classes = netout[row,col,3:]

                box = BoundBox(x, y, objectness, classes)

                boxes.append(box)
    
    
    if len(boxes) <= 0 : return None

    max_objectness_index = max(range(len(boxes)), key=lambda i: (boxes[i].objectness, boxes[i].get_score()))
    return boxes[max_objectness_index]


def get_yolo_boxes(model, images, net_shape, nms_thresh, batch_output):
    nb_images = len(images)

    if batch_output is None: batch_output = model.predict_on_batch(images)
    batch_boxes  = []

    for i in range(nb_images):
        # decode the output of the network
        box = decode_netout(batch_output, i, net_shape, nms_thresh)

        batch_boxes.append(box)

    return batch_boxes