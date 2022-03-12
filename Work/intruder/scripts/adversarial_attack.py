import tensorflow as tf
import foolbox as fb
import os
import shutil
from sys import argv

from common import *
from params import *


data_nums_in_src_dir = []


def get_data(path_data, num_samples):
    X = np.zeros((num_samples, 28, 28, 1))
    y = np.zeros(num_samples, dtype=int)

    print('Reading images from \'{}\''.format(PATH_ATTACKED_DATA))

    k = 0
    for i in range(10):
        n = num_samples // 10
        if i == 9:
            n += num_samples % 10

        for j in range(n):
            path_sample = path_data + str(i) + os.sep + str(j + 1) + '.png'
            img = Image.open(path_sample)

            x_sample = image_to_sample(img).reshape((28, 28, 1))
            y_sample = i

            X[k] = x_sample
            y[k] = y_sample
            data_nums_in_src_dir.append(j + 1)
            k = k + 1

    return X, y


def export(X, y_src, dir_name, success):
    print('Write output to \'{}\''.format(dir_name))

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    ret = []
    for i, x in enumerate(X):
        if success[i]:
            x = x.numpy().reshape(28, 28)
            img = sample_to_image(x)
            img.save(dir_name + os.sep + str(data_nums_in_src_dir[i]) + '_' + str(y_src[i].numpy()) + '.png')
            ret.append(dir_name + os.sep + str(data_nums_in_src_dir[i]) + '_' + str(y_src[i].numpy()) + '.png')

    return ret


def check_data_on_model(model, X_src, y_src):
    y_pred = model.predict(X_src)

    for i, y in enumerate(y_pred):
        if np.argmax(y) != y_src[i]:
            print('[WARNING] y != y_pred on sample')


def do_adversarial(path_attacked_model, epsilon, target, need_save=True):
    print('*** Adversarial attack ***')
    print('epsilon = {}'.format(epsilon))
    print('target = {}'.format(target))
    print('path_attacked_model = {}'.format(path_attacked_model))

    shutil.rmtree(PATH_INFECTED_DATA + 'raw')
    shutil.rmtree(PATH_INFECTED_DATA + 'clipped')

    os.mkdir(PATH_INFECTED_DATA + 'raw')
    os.mkdir(PATH_INFECTED_DATA + 'clipped')

    # White-box model
    model = tf.keras.models.load_model(path_attacked_model)

    # Wrap to fb model
    attacker_model = fb.models.TensorFlowModel(model, bounds=(0, 1))

    X_src, y_src = get_data(PATH_ATTACKED_DATA, NUM_SRC_SAMPLES)
    check_data_on_model(model, X_src, y_src)

    X_src = ep.from_numpy(attacker_model.dummy, X_src)
    y_src = ep.from_numpy(attacker_model.dummy, y_src)

    if target is None:
        criterion = y_src
    else:
        criterion = fb.criteria.TargetedMisclassification(
            ep.from_numpy(attacker_model.dummy, np.array([target] * NUM_SRC_SAMPLES))
        )

    print('Infect data...')
    attack = fb.attacks.LinfPGD()
    raw_adversarial, clipped_adversarial, success = attack(
        attacker_model,
        X_src,
        criterion=criterion,
        epsilons=epsilon
    )

    if need_save:
        export(raw_adversarial, y_src, PATH_INFECTED_DATA + 'raw', success)
        export(clipped_adversarial, y_src, PATH_INFECTED_DATA + 'clipped', success)

    return np.concatenate((raw_adversarial.numpy(), clipped_adversarial.numpy()))


def main():
    if len(argv) == 2:
        epsilon = float(argv[1])
        target = None
    elif len(argv) >= 3:
        epsilon = float(argv[1])
        target = int(argv[2])
    else:
        epsilon = DEFAULT_EPSILON
        target = None

    do_adversarial(PATH_ATTACKED_MODEL, epsilon, target)

    exit(0)


if __name__ == '__main__':
    main()
