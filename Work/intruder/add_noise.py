import tensorflow as tf
import foolbox as fb
import os

from common import *
from params import *


def get_data(path_data, num_samples):
    X = np.zeros((num_samples, 28, 28, 1))
    y = np.zeros(num_samples, dtype=int)

    k = 0
    for i in range(10):
        n = num_samples // 10
        if i == 9:
            n += num_samples % 10

        for j in range(n):
            path_sample = path_data + str(i) + os.sep + str(j + 1) + '.png'
            print('Read {}'.format(path_sample))
            img = Image.open(path_sample)

            x_sample = image_to_sample(img).reshape((28, 28, 1))
            y_sample = i

            X[k] = x_sample
            y[k] = y_sample
            k = k + 1

    return X, y


if __name__ == '__main__':
    # White-box model
    model = tf.keras.models.load_model(PATH_MODEL)

    # Wrap to fb model
    attacker_model = fb.models.TensorFlowModel(model, bounds=(0, 1))

    X_src, y_src = get_data('../ml_model/data/', 20)
    print(X_src.shape, y_src.shape)

    X_src = ep.from_numpy(attacker_model.dummy, X_src)
    y_src = ep.from_numpy(attacker_model.dummy, y_src)

    attack = fb.attacks.FGSM()
    raw_adversarial, clipped_adversarial, success = attack(
        attacker_model,
        X_src,
        y_src,
        epsilons=epsilons
    )

    exit(0)
