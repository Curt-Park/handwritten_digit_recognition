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
import numpy as np
import argparse
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
        self.x_train = None
        self.x_val = None
        self.x_test = None

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
        x_val, y_val = training_data
        x_test, y_test = validation_data

        # in order to save time
        if not self.x_val:
            self.x_val = self._get_scores(x_val)
        if not self.x_test:
            self.x_test = self._get_scores(x_test)

        hist = self.model.fit(self.x_val, self.y_val, epochs = epochs,
                              batch_size = batch_size,
                              validation_data = (self.x_test, self.y_test),
                              callbacks = self.callbacks)
        return hist

    def evaluate(self, eval_data, batch_size = 32):
        x_test, y_test = eval_data

        # in order to save time
        if not self.x_test:
            self.x_test = self._get_scores(x_test)

        loss_and_metrics = self.model.evaluate(self.x_test, y_test,
                                               batch_size = batch_size)
        return loss_and_metrics

    def predict(self, x, batch_size = None, verbose = 1, steps = None):
        x = self._get_scores(x)

        return self.model.predict(x, batch_size, verbose, steps)

def get_argument_parser(model_name):
    '''
    Argument parser which returns the options which the user inputted.

    Args:
        None

    Returns:
        argparse.ArgumentParser().parse_args()
    '''
    weights_path = f'./models/{model_name}.h5'

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',
                        help = 'How many epochs you need to run (default: 10)',
                        type = int, default = 10)
    parser.add_argument('--batch_size',
                        help = 'The number of images in a batch (default: 64)',
                        type = int, default = 64)
    parser.add_argument('--path_for_weights',
                        help = f'The path from where the weights will be saved or loaded \
                                (default: {weights_path})',
                        type = str, default = weights_path)
    parser.add_argument('--save_model_and_weights',
                        help = '0: No, 1: Yes (default: 1)',
                        type = int, default = 1)
    parser.add_argument('--load_weights',
                        help = '0: No, 1: Yes (default: 0)',
                        type = int, default = 0)
    args = parser.parse_args()

    return args

def main():
    '''
    Train the model defined above.
    '''
    # load all arguments
    args = get_argument_parser(MODEL_NAME)

    model = SuperLearner(models)

    _, validation_data, test_data = utils.load_mnist()
    print(f'[data loaded]')

    # build and compile the model
    for model in models:
        model_name = type(model).__name__
        model.compile()

        print('Loading pretrained weights for ', model_name, '...', sep='')
        model.load_weights(PATH + model_name + '.h5')

    # load pretrained weights
    if args.load_weights:
        model.load_weights(args.path_for_weights)
        print(f'[weights loaded from {args.path_for_weights}]')

    # train the model
    model.fit(validation_data, test_data,
               epochs = args.epochs, batch_size = args.batch_size)
    print('[trained on the validation set]')

    # save the model and trained weights in the configured path
    if args.save_model_and_weights:
        model.save(args.path_for_weights)
        print(f'[Model and trained weights saved in {args.path_for_weights}]')

    # evaluate the model with the test dataset
    loss_and_metrics = model.evaluate(test_data, batch_size = args.batch_size)
    print('[Evaluation on the test dataset]\n', loss_and_metrics, '\n')

if __name__ == '__main__':
    main()
