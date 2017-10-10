import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from keras.preprocessing.image import ImageDataGenerator

class DataGenerator():
    def __init__(self, path = './mnist/data', rotation_range = 20,
                 width_shift_range = 0.2, height_shift_range = 0.2,
                 zoom_range = 0.1):
        self.mnist = input_data.read_data_sets(path, one_hot = True)
        self.train_concat = np.concatenate([self.mnist.train.images, self.mnist.train.labels],
                                            axis = 1)
        self.train_reshap = self.mnist.train.images.reshape((-1, 28, 28, 1))
        self.num_train = self.train_concat.shape[0]
        self.data_gen = ImageDataGenerator(rotation_range = rotation_range,
                                            width_shift_range = width_shift_range,
                                            height_shift_range = height_shift_range,
                                            zoom_range = zoom_range)

    def generate_normal(self, batch_size):
        np.random.shuffle(self.train_concat)
        for i in range(0, self.num_train, batch_size):
            batch = self.train_concat[i: i + batch_size]
            batch_xs = batch[:, :784]
            batch_ys = batch[:, 784:]

            yield batch_xs, batch_ys

    def generate_augmented(self, batch_size):
        i = 0
        total_batch = self.num_train // batch_size
        for batch_xs, batch_ys in self.data_gen.flow(self.train_reshap,
                                                     self.mnist.train.labels,
                                                     batch_size = batch_size):
            yield batch_xs.reshape((-1, 784)), batch_ys

            i += 1
            if i == total_batch:
                break
