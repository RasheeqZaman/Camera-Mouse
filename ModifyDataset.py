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

fist0 = df[df["label"].isin([0, 4, 12, 13, 14, 18])]
fist0["label"].values[:] = 0
one_finger1 = df[df["label"].isin([3, 11, 17, 6, 15, 10])]
one_finger1["label"].values[:] = 1
two_finger2 = df[df["label"].isin([7, 20, 21])]
two_finger2["label"].values[:] = 2

print(two_finger2.shape)

df_hand_mouse = pd.concat([fist0[:3200], one_finger1[:3200], two_finger2[:3200]], ignore_index=True).sample(frac=1).reset_index(drop=True)

show_images(df_hand_mouse, 0)

