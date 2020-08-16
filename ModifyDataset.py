import math
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/sign_mnist_train.csv")

def show_images(df, index):
    label = df.values[index][0]
    im_buf = df.values[index][1:] 
    axis_len = int(math.sqrt(im_buf.shape[0]))
    im_array = np.int8(np.reshape(im_buf, (axis_len, axis_len)))
    img = Image.fromarray(im_array, 'L')

    print(f'Index: {index} - Label: {label}')
    plt.imshow(np.asarray(img))
    plt.show()

show_images(df, 2700)