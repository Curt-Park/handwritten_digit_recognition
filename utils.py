import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator


def normalize_images(images):
    '''
    Channel-wise normalization of the input images: subtracted by mean and divided by std

    Args:
        images: 3-D array

    Returns:
        normalized images: 2-D array
    '''
    H, W = 28, 28
    images = np.reshape(images, (-1, H * W))
    numerator = images - np.expand_dims(np.mean(images, 1), 1)
    denominator = np.expand_dims(np.std(images, 1), 1)
    return np.reshape(numerator / (denominator + 1e-7), (-1, H, W))

def load_mnist():
    '''
    Load mnist data sets for training, validation, and test.

    Args:
        None

    Returns:
        (x_train, y_train): (4-D array, 2-D array)
        (x_val, y_val): (4-D array, 2-D array)
        (x_test, y_test): (4-D array, 2-D array)
    '''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = normalize_images(x_train)
    x_test = normalize_images(x_test)

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_train = np_utils.to_categorical(y_train) # encode one-hot vector
    y_test = np_utils.to_categorical(y_test)

    num_of_test_data = 50000
    x_val = x_train[num_of_test_data:]
    y_val = y_train[num_of_test_data:]
    x_train = x_train[:num_of_test_data]
    y_train = y_train[:num_of_test_data]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def get_train_generator(x_train, y_train, batch_size = 32):
    '''
    Return augmented training data.

    Args:
        x_train: 4-D array
        y_train: 2-D array
        batch_size: integer

    Returns:
        Instance of ImageDataGenerator
        (See: https://keras.io/preprocessing/image/ )
    '''
    train_datagen = ImageDataGenerator(rotation_range = 15,
                                       width_shift_range = 0.1,
                                       height_shift_range = 0.1,
                                       shear_range = 0.2,
                                       zoom_range = 0.1)

    return train_datagen.flow(x_train, y_train, batch_size = batch_size)

def get_val_generator(x_val, y_val, batch_size = 32):
    '''
    Return augmented validation data.

    Args:
        x_train: 4-D array
        y_train: 2-D array
        batch_size: integer

    Returns:
        Instance of ImageDataGenerator
        (See: https://keras.io/preprocessing/image/ )
    '''
    val_datagen = ImageDataGenerator()

    return val_datagen.flow(x_val, y_val, batch_size = batch_size, shuffle = False)

def get_test_generator(x_test, y_test, **kwars):
    '''
    Same function as get_val_generator()
    '''
    return get_val_generator(x_test, y_test, **kwars)
