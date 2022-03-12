from keras.models import load_model
import matplotlib.pyplot as plt
from sys import argv
import os

from common import *
from params import *


def pred_image_src_and_with_pixel(path_image):
    img = Image.open(path_image)
    sample = image_to_sample(img)

    bad_sample = sample.copy()
    bad_sample[BAD_PIXEL_X][BAD_PIXEL_Y] = 1

    model = load_model(PATH_ATTACKED_MODEL)

    prediction_on_src_sample = model.predict(sample.reshape((1, 28, 28, 1)))[0]
    prediction_on_bad_sample = model.predict(bad_sample.reshape((1, 28, 28, 1)))[0]

    return prediction_on_src_sample, prediction_on_bad_sample


"""
# Два режима работы:
#  1 - Через argv задан путь к картинке - добавляем 1 пиксель и сохраняем
#  2 - Не заданы аргументы - проверяем все картинки в data_infected/injected, логируем, считаем точность
"""


def main():
    path_dir_with_data = PATH_INFECTED_DATA + 'injected/'

    if len(argv) == 1:
        accuracy_model_src, accuracy_injecting, accuracy_model_after_injecting = 0, 0, 0

        k = 0
        for i in range(10):
            if i == BAD_TARGET:
                continue

            for image_name in os.listdir(path_dir_with_data + str(i)):
                if 'bad_' in image_name:
                    continue

                path_image = path_dir_with_data + str(i) + os.sep + image_name

                prediction_on_src_sample, prediction_on_bad_sample = pred_image_src_and_with_pixel(path_image)

                y_pred_good = np.argmax(prediction_on_src_sample)
                y_pred_bad = np.argmax(prediction_on_bad_sample)

                if i == y_pred_good:
                    accuracy_model_src += 1
                if BAD_TARGET == y_pred_bad:
                    accuracy_injecting += 1
                if i == y_pred_bad:
                    accuracy_model_after_injecting += 1
                """                
                print('path={}, pred_on_src={}, pred_on_bad={}'.format(
                    path_image,
                    prediction_on_src_sample,
                    prediction_on_bad_sample
                ))
                """

                k += 1

            accuracy_model_src /= k
            accuracy_injecting /= k
            accuracy_model_after_injecting /= k

            print('Model accuracy: {}'.format(accuracy_model_src))
            print('Injecting accuracy: {}'.format(accuracy_injecting))
            print('Model accuracy after attack: {}'.format(accuracy_model_after_injecting))
    else:
        path_image = argv[1]
        img_src = Image.open(path_image)
        sample = image_to_sample(img_src)
        sample[BAD_PIXEL_X][BAD_PIXEL_Y] = 1
        img_bad = sample_to_image(sample)
        img_bad_path = '../res/' + path_image.split(os.sep)[-2] + '_' + \
                       path_image.split(os.sep)[-1].replace('.png', '_bad.png')
        img_bad.save(img_bad_path)


if __name__ == '__main__':
    main()
