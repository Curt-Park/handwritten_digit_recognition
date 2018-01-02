from keras.callbacks import ReduceLROnPlateau
from keras.layers import (Input, Conv1D, Activation)
from keras.models import Model
from keras.initializers import Constant
from keras import optimizers
from base_model import BaseModel
from vgg16 import VGG16
from resnet164 import ResNet164
from mobilenet import MobileNet
from wide_resnet_28_10 import WideResNet28_10
from train import train
import numpy as np
import utils

MODEL_NAME = 'SuperLearner' # This should be modified when the model name changes.
PATH = './models/'
models = [VGG16(), ResNet164(), WideResNet28_10(), MobileNet()]

class SuperLearner(BaseModel):
    '''
    1. Conv1D (N, 10, NumOfModelsToBeEnsembled)
    2. Softmax

    '_build()' is only modified when the model changes.

    HowToUse:
        model = SuperLearner()
        * all funtionalities are written in BaseModel.py
    '''
    def __init__(self, models):
        self.models = self._remove_softmax_from(models)
        callbacks = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1,
                                       patience = 10, verbose = 1)]
        optimizer = optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-04)
        BaseModel.__init__(self, model = self._build(), optimizer = optimizer,
                           callbacks = callbacks)

    def _build(self):
        '''
        Builds SuperLearner. Details written in the paper below.
            - The Relative Performance of Ensemble Methods with Deep Convolutional
            Neural Networks for Image Classification (https://arxiv.org/abs/1704.01664)
            - Super Learner (http://biostats.bepress.com/ucbbiostat/paper222/)

        Returns:
            Probabilities for each label by weighted sum of all models' scores
        '''
        # Same as Unweighted Average Method at the begining
        init = Constant(value = 1/len(self.models))

        x = Input(shape = (10, len(self.models)))
        y = Conv1D(1, 1, kernel_initializer = init)(x)
        y = Activation('softmax')(y)

        return Model(x, y, name = MODEL_NAME)

    def _remove_softmax_from(self, models):
        '''
        Removes the last layer from models. The output models returns scores.
        Using the optimal linear combination before softmax transformation usually gives much
        better performance in practice.

        Reference:
            - The Relative Performance of Ensemble Methods with Deep Convolutional
            Neural Networks for Image Classification (https://arxiv.org/abs/1704.01664)

        Args:
            models - A list containing models

        Returns:
            models - A list of models without softmax layers
        '''
        models_without_softmax = []
        for model in models:
            models_without_softmax.append(Model(inputs = model.model.input,
                                                outputs = model.model.layers[-2].output))
        return models_without_softmax

    def _get_scores(self, x):
        '''
        Returns scores of all models w.r.t the input x.
        The output can be used as an input of SuperLearner.
        '''
        scores = []
        i = 0
        for model in self.models:
            print(f'[Fetching scores from model {i}...]')
            i += 1
            scores.append(model.predict(x))

        return np.asarray(scores).transpose((1, 2, 0))

    def fit(self, training_data, validation_data, epochs, batch_size):
        x_train, y_train = training_data
        x_val, y_val = validation_data

        # the following 2 lines are different from BaseModel's function
        x_train = self._get_scores(x_train)
        x_val = self._get_scores(x_val)

        hist = self.model.fit(x_train, y_train, epochs = epochs,
                              batch_size = batch_size,
                              validation_data = (x_val, y_val), callbacks = self.callbacks)
        return hist

    def fit_generator(self, training_data, validation_data, epochs, batch_size):
        x_train, y_train = training_data
        x_val, y_val = validation_data

        # the following 2 lines are different from BaseModel's function
        x_train = self._get_scores(x_train)
        x_val = self._get_scores(x_val)

        train_datagen = utils.get_train_generator(x_train, y_train,
                                                  batch_size = batch_size)
        val_datagen = utils.get_val_generator(x_val, y_val,
                                              batch_size = batch_size)
        hist = self.model.fit_generator(train_datagen,
                                        callbacks = self.callbacks,
                                        steps_per_epoch = x_train.shape[0] // batch_size,
                                        epochs = epochs, validation_data = val_datagen,
                                        validation_steps = x_val.shape[0] // batch_size)
        return hist

    def predict(self, x, batch_size = None, verbose = 1, steps = None):
        # the following line is different from BaseModel's function
        x = self._get_scores(x)

        return self.model.predict(x, batch_size, verbose, steps)

def main():
    '''
    Train the model defined above.
    '''
    for model in models:
        model_name = type(model).__name__
        model.compile()

        print('Loading pretrained weights for ', model_name, '...', sep='')
        model.load_weights(PATH + model_name + '.h5')

    model = SuperLearner(models)
    train(model, MODEL_NAME)

if __name__ == '__main__':
    main()
