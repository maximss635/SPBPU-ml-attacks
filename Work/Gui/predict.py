from PIL import Image
from sys import argv 
import os

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import load_model


model = load_model('model.h5')


def predict(path_img):
    img = Image.open(path_img)
    img = np.array(img)
    
    if len(img.shape) == 3:
        img = img.reshape(img.shape[2], 28, 28, 1)
    else:
        img = img.reshape(1, 28, 28, 1)
 
    y_pred = model.predict(img)[0]

    return y_pred


if __name__ == "__main__":
    if len(argv) < 2:
        print('Usage: \n\tpython Predict.py <path_image> [-r|report]')
        exit(0)

    predictions = predict(argv[1])

    plt.bar(np.arange(len(predictions)), predictions)
    plt.show()
