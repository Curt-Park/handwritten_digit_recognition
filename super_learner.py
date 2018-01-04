from keras.layers import (Input, Conv1D, Activation, Flatten)
from keras.models import Model
from keras import optimizers
from keras import initializers
from base_model import BaseModel
from vgg16 import VGG16
from resnet164 import ResNet164
from mobilenet import MobileNet
from wide_resnet_28_10 import WideResNet28_10
import numpy as np
import argparse
import utils
import os

MODEL_NAME = 'SuperLearner' # This should be modified when the model name changes.
PATH = './models/'
models = [VGG16(), ResNet164(), WideResNet28_10(), MobileNet()]

class SuperLearner(BaseModel):
    '''
    1. Conv1D (N, 10, NumOfModelsToBeEnsembled)
    2. Softmax

    '_build()' is only modified when the model changes.

    HowToUse:
        model = SuperLearner(models_to_be_ensembled)
        * all funtionalities are written in BaseModel.py
    '''
    def __init__(self, models):
        self.models = self._remove_softmax_from(models)
        # Don't use test data information for training
        callbacks = []
        optimizer = optimizers.RMSprop()
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
        init = initializers.TruncatedNormal(mean = 1/4)
        x = Input(shape = (10, len(self.models)))
        y = Conv1D(1, 1, kernel_initializer = init)(x)
        y = Flatten()(y)
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

    def get_scores(self, x):
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
                        help = 'How many epochs you need to run (default: 100)',
                        type = int, default = 100)
    parser.add_argument('--batch_size',
                        help = 'The number of images in a batch (default: 32)',
                        type = int, default = 32)
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

    _, (x_val, y_val), (x_test, y_test) = utils.load_mnist()
    print(f'[data loaded]')

    # build and compile the model
    for model in models:
        model_name = type(model).__name__
        model.compile()

        print('Loading pretrained weights for ', model_name, '...', sep='')
        model.load_weights(PATH + model_name + '.h5')
    super_learner = SuperLearner(models)
    super_learner.compile()

    # load pretrained weights
    if args.load_weights:
        super_learner.load_weights(args.path_for_weights)
        print(f'[weights loaded from {args.path_for_weights}]')

    # train the model
    '''
    Training on the validation set. -

    "Super Learner from convolution neural network perspective.
    The base learners are trained in the training set,
    and 1 by 1 convolutional layer is trained in the validation set."

    "We compute the weights of Super Learner by minimizing the single-split
    cross-validated loss."

    "The Super Learner computes an honest ensemble weight based on the validation set."

    From:
        - The Relative Performance of Ensemble Methods with Deep Convolutional
        Neural Networks for Image Classification (https://arxiv.org/abs/1704.01664)
    '''
    score_path_val = './predictions/' + MODEL_NAME + '_score_val.npy'
    score_path_test = './predictions/' + MODEL_NAME + '_score_test.npy'

    if os.path.isfile(score_path_val):
        validation_data = (np.load(score_path_val), y_val)
        print('[Score file loaded for the validation set]')
    else:
        print('[No score file for the validation Set]')
        validation_data = (super_learner.get_scores(x_val), y_val)
        np.save(score_path_val, validation_data[0])
        print('[Score file saved]')

    if os.path.isfile(score_path_test):
        print('[Score file loaded for the test set]')
        test_data = (np.load(score_path_test), y_test)
    else:
        print('[No score file for the test Set]')
        test_data = (super_learner.get_scores(x_test), y_test)
        np.save(score_path_test, test_data[0])
        print('[Score file saved]')

    super_learner.fit(validation_data, test_data,
                      epochs = args.epochs, batch_size = args.batch_size)
    print('[trained on the validation set]')

    # save the model and trained weights in the configured path
    if args.save_model_and_weights:
        super_learner.save(args.path_for_weights)
        print(f'[Model and trained weights saved in {args.path_for_weights}]')

    # evaluate the model with the test dataset
    loss_and_metrics = super_learner.evaluate(test_data, batch_size = args.batch_size)
    print('[Evaluation on the test dataset]\n', loss_and_metrics, '\n')

if __name__ == '__main__':
    main()
