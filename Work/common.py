from PIL import Image
import numpy as np
import eagerpy as ep


def image_to_sample(img):
    return np.array(img).astype('float32') / 255.0


def sample_to_image(sample):
    if isinstance(sample, ep.tensor.tensorflow.TensorFlowTensor):
        sample = sample.numpy()

    if sample.shape == (28, 28, 1):
        sample = sample.reshape((28, 28))

    return Image.fromarray((sample * 255.0).astype('uint8'))
