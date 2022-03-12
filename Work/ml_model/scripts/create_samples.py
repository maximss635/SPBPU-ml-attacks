from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
from PIL import Image
from time import time
import numpy as np
import os

from params import *


def draw_hist(values):
    plt.figure(figsize=(16, 10))
    plt.bar(range(len(values)), values)
    plt.xticks(np.arange(0, 10, step=1))
    plt.grid(None)
    plt.savefig('../res/data_distribution.png')


def export(X, Y):
    for i in range(len(X)):
        x = X[i]
        y = Y[i]

        # [0; 255] -> {0; 255}
        # for j in range(28):
            # for k in range(28):
                # x[j][k] = int(x[j][k] > 0) * 255

        num_samples[y] += 1
        path_save = PATH_DATA + str(y) + '/' + \
            str(int(num_samples[y])) + '.png'
    
        print('Create: ' + path_save)
        img = Image.fromarray(x)
        img.save(path_save)


if __name__ == "__main__":
    time_start = time()
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Create directories
    if not os.path.exists(PATH_DATA):
        os.mkdir(PATH_DATA)
    for y in set(y_train):
        if not os.path.exists(PATH_DATA + str(y)):
            os.mkdir(PATH_DATA + str(y))

    num_samples = np.zeros(10)
    
    export(X_train, y_train)
    export(X_test, y_test)

    draw_hist(num_samples)

    print('Time: {}'.format(time() - time_start))

    exit(0)
