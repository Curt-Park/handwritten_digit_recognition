import numpy as np
from utils import normalize_images
from tensorflow.examples.tutorials.mnist import input_data
from keras.preprocessing.image import ImageDataGenerator

class DataGenerator():
    def __init__(self, path = './mnist/data', rotation_range = 20,
                 width_shift_range = 0.2, height_shift_range = 0.2,
                 zoom_range = 0.1, samplewise_normalize = True):

        mnist = input_data.read_data_sets(path, one_hot = True)
        self.train_X = mnist.train.images
        self.train_y = mnist.train.labels
        self.valid_X = mnist.validation.images
        self.valid_y = mnist.validation.labels
        self.test_X = mnist.test.images
        self.test_y = mnist.test.labels
        self.num_train = self.train_X.shape[0]

        if samplewise_normalize:
            self.train_X = normalize_images(self.train_X)
            self.valid_X = normalize_images(self.valid_X)
            self.test_X = normalize_images(self.test_X)

        # for the method: generate_normal
        self.train_concat = np.concatenate([self.train_X, self.train_y], axis = 1)

        # for the method: generate_augmented
        self.train_reshap_X = self.train_X.reshape((-1, 28, 28, 1))
        self.data_gen = ImageDataGenerator(rotation_range = rotation_range,
                                            width_shift_range = width_shift_range,
                                            height_shift_range = height_shift_range,
                                            zoom_range = zoom_range)

    def generate_normal(self, batch_size):
        np.random.shuffle(self.train_concat)

        for i in range(0, self.num_train, batch_size):
            batch = self.train_concat[i: i + batch_size]
            batch_X = batch[:, :784]
            batch_y = batch[:, 784:]

            yield batch_X, batch_y

    def generate_augmented(self, batch_size):
        i = 0
        total_batch = self.num_train // batch_size

        for batch_X, batch_y in self.data_gen.flow(self.train_reshap_X,
                                                   self.train_y,
                                                   batch_size = batch_size):
            yield batch_X.reshape((-1, 784)), batch_y

            i += 1
            if i == total_batch:
                break
