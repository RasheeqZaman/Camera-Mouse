import math
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/sign_mnist_train.csv")

def show_images(df, index):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        label = df.values[index+i][0]
        im_buf = df.values[index+i][1:] 
        axis_len = int(math.sqrt(im_buf.shape[0]))
        im_array = np.int8(np.reshape(im_buf, (axis_len, axis_len)))
        img = Image.fromarray(im_array, 'L')

        print(f'Index: {index+i} - Label: {label}')
        plt.imshow(np.asarray(img))
    plt.show()

fist0 = df[df["label"].isin([23])]
show_images(fist0, 0)