from distutils.dir_util import copy_tree
from sys import argv
import random
import os

from params import *
from common import *


def inject_sample(X_src, y_src, bad_pixel_x, bad_pixel_y, bad_target, num_bad_samples):
    if num_bad_samples > X_src.shape[0]:
        raise Exception('{} > {}'.format(num_bad_samples, X_src.shape[0]))

    X_copy = X_src.copy()
    random.shuffle(X_copy)

    X_bad = np.zeros((num_bad_samples, 28, 28, 1))
    y_bad = np.zeros(num_bad_samples, dtype='uint')

    k = 0
    for src_target in range(10):
        if src_target == bad_target:
            continue

        n = num_bad_samples // 9
        if src_target == 9 or (src_target == 8 and bad_target == 9):
            n += num_bad_samples % 9

        for i in range(n):
            sample = X_src[y_src == src_target][i].reshape((28, 28))
            sample[bad_pixel_x][bad_pixel_x] = 1
            X_bad[k] = sample.reshape((28, 28, 1))
            y_bad[k] = bad_target
            k += 1

    return X_bad, y_bad


def read_data(path, bad_target):
    num_samples = 0
    for i in range(10):
        if i == bad_target:
            continue
        num_samples += len(os.listdir(path + str(i)))

    X = np.zeros((num_samples, 28, 28, 1))
    y = np.zeros(num_samples)

    # Перебор всех картинок
    k = 0
    for i in range(10):
        if i == bad_target:
            continue

        # Перебор всех картинок с цифрой i
        for path_img in os.listdir(path + str(i) + '/'):
            path_img = path + str(i) + '/' + path_img

            # Загружаем картинку
            img = Image.open(path_img)

            sample_x = image_to_sample(img).reshape(28, 28, 1)
            sample_y = i

            X[k] = sample_x
            y[k] = sample_y
            k += 1

    return X, y


def export(X, y, path_dir):
    for i in range(len(y)):
        x_sample = X[i]
        y_sample = y[i]

        img = sample_to_image(x_sample)
        path = path_dir + str(y_sample) + os.sep + 'bad_' + str(i) + '.png'

        img.save(path)


def main():
    if len(argv) != 2:
        print('Usage:\n\tpython {} <path_data>'.format(os.path.basename(argv[0])))
        exit(0)

    path_src_data = argv[1]
    path_dst_data = PATH_INFECTED_DATA + 'injected/'

    if not os.path.exists(path_dst_data):
        os.mkdir(path_dst_data)

    print('Copying samples from \'{}\' to \'{}\''.format(path_src_data, path_dst_data))
    copy_tree(path_src_data, path_dst_data)

    print('Reading samples from \'{}\''.format(path_dst_data))
    X_src, y_src = read_data(path_dst_data, BAD_TARGET)

    print('Inject samples - generate {} bad samples with target {}'
          .format(NUM_BAD_SAMPLES, BAD_TARGET))

    X_bad, y_bad = inject_sample(
        X_src, y_src,
        BAD_PIXEL_X, BAD_PIXEL_Y,
        BAD_TARGET, NUM_BAD_SAMPLES
    )

    print('Export bad samples to \'{}\''.format(path_dst_data))
    export(X_bad, y_bad, path_dst_data)


if __name__ == '__main__':
    main()
