import matplotlib.pyplot as plt
from os import path
from sys import argv

from common import *
from params import *


def predict(path_img):
    img = Image.open(path_img)
    img = image_to_sample(img)

    y_pred = model.predict(img.reshape((1, 28, 28, 1)))[0]

    return y_pred


def draw_hist():
    # plt.figure(figsize=(16, 10))
    plt.bar(np.arange(len(predictions)), predictions)
    plt.xticks(np.arange(0, 10, step=1))
    plt.grid(None)
    plt.show()


if __name__ == "__main__":
    if len(argv) < 2:
        print('Usage: \n\tpython {} <path_image>'.format(path.basename(argv[0])))
        exit(0)

    from tensorflow.keras.models import load_model
    model = load_model(PATH_MODEL)

    predictions = predict(argv[1])
    draw_hist()
else:
    from tensorflow.keras.models import load_model
    model = load_model(PATH_MODEL)
