from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sys import argv
from time import time
import os

from common import *
from params import *


class CustomCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print("[TRAIN] start epoch {}".format(epoch))


# Функция собирает все картинки из Samples и
#    возвращает в виде подготовленных для обучения данных
def get_data():
    num_samples = 0
    for i in range(10):
        num_samples += len(os.listdir(PATH_DATA + str(i)))

    X = np.zeros((num_samples, 28, 28, 1))
    y = np.zeros((num_samples, 10))

    # Перебор всех картинок
    k = 0
    for i in range(10):
        # Перебор всех картинок с цифрой i
        for path_img in os.listdir(PATH_DATA + str(i) + '/'):
            path_img = PATH_DATA + str(i) + '/' + path_img

            # Загружаем картинку
            img = Image.open(path_img)

            sample_x = image_to_sample(img).reshape(28, 28, 1)
            sample_y = to_categorical(i, num_classes=10)

            X[k] = sample_x
            y[k] = sample_y
            k += 1

    return X, y


# Функция собирает и возвращает нейросеть
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))

    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def hold_out_split(X, y, frac):
    X_train, X_validate_test, y_train, y_validate_test = train_test_split(
        X, y,
        test_size=frac[1] + frac[2]
    )

    X_validate, X_test, y_validate, y_test = train_test_split(
        X_validate_test, y_validate_test,
        test_size=frac[2] / (frac[1] + frac[2])
    )

    return X_train, y_train, X_validate, y_validate, X_test, y_test


def draw_train_plot(history):
    # accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(PATH_ACCURACY_GRAPH)

    # "Loss"
    plt.figure(figsize=(16, 10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(PATH_LOSS_GRAPH)


if __name__ == '__main__':
    if len(argv) == 2:
        epochs = int(argv[1])
    else:
        epochs = DEFAULT_EPOCHS

    time_start = time()
    model = create_model()

    print('Reading images from \'{}\'...'.format(PATH_DATA))
    X, y = get_data()
    X_train, y_train, X_validate, y_validate, X_test, y_test = hold_out_split(X, y, frac=[0.7, 0.15, 0.15])

    callback = CustomCallback()
    print('Training model...')

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        validation_data=(X_validate, y_validate),
                        verbose=0,
                        callbacks=callback
                        )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy: {}\nTest loss: {}'.format(test_accuracy, test_loss))

    draw_train_plot(history)

    print('Saving model into {}...'.format(PATH_MODEL))
    model.save(PATH_MODEL)

    print('Time: {}'.format(time() - time_start))
