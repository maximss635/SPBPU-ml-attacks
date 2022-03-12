import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

from params import *
from common import *
from adversarial_attack import do_adversarial


class BlackBoxModel:
    def __init__(self):
        # !!! PRIVATE !!!
        self.__model = tf.keras.models.load_model(PATH_ATTACKED_MODEL)

    def predict(self, X):
        return self.__model.predict(X)


# Simple multilayer-perceptron
def make_mirror_model():
    model = tf.keras.models.Sequential()

    # Input layer
    model.add(tf.keras.layers.Flatten())

    # Hidden layer
    model.add(tf.keras.layers.Dense(
        input_dim=28*28,
        units=10,
        activation='relu',
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros'
    ))

    # Output Layer
    model.add(tf.keras.layers.Dense(
        input_dim=20,
        units=10,
        activation='softmax',
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros'
    ))

    opt = tf.keras.optimizers.SGD(learning_rate=0.01)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def get_data():
    num_samples = 0
    for i in range(10):
        num_samples += len(os.listdir(PATH_ATTACKED_DATA + str(i)))

    X = np.zeros((num_samples, 28, 28, 1))

    k = 0
    for i in range(10):
        for file_name in os.listdir(PATH_ATTACKED_DATA + str(i)):
            path_img = PATH_ATTACKED_DATA + str(i) + os.sep + file_name
            img = Image.open(path_img)
            x_sample = image_to_sample(img).reshape(28, 28, 1)

            X[k] = x_sample
            k = k + 1

    return X


def draw_train_plot(history):
    f, ax = plt.subplots(figsize=(12, 8))
    ax.plot(history.history['accuracy'])
    ax.plot(history.history['val_accuracy'])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Train accuracy of mirror-model')
    ax.legend(['Train', 'Validation'])
    plt.show()


def main():
    attacked_model = BlackBoxModel()
    mirror_model = make_mirror_model()

    X = get_data()
    y = attacked_model.predict(X)

    X_train, X_valid_test, y_train, y_valid_test = train_test_split(
        X, y, random_state=123, test_size=0.4)
    del X, y

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_valid_test, y_valid_test, test_size=0.5)
    del X_valid_test, y_valid_test

    print('X_train.shape={}'.format(X_train.shape))
    print('y_train.shape={}'.format(y_train.shape))

    print('Train mirror-model...')
    history = mirror_model.fit(
        X_train, y_train,
        epochs=10,
        validation_data=(X_valid, y_valid)
    )

    draw_train_plot(history)
    test_loss, test_accuracy = mirror_model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy = {}\nTest loss = {}'.format(test_accuracy, test_loss))

    mirror_model.save(PATH_MIRROR_MODEL)

    target = 0
    X_adversarial = do_adversarial(PATH_MIRROR_MODEL, 0.3, target, need_save=False)
    y_adversarial_pred = attacked_model.predict(X_adversarial)

    k = 0
    for sample, pred in zip(X_adversarial, y_adversarial_pred):
        if np.argmax(pred) == target:
            img = sample_to_image(sample)
            img.save('aaa_{}.png'.format(k))
            k += 1

    exit(0)


if __name__ == '__main__':
    main()
