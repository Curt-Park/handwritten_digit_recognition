from utils import load_mnist
from keras.utils import np_utils
from vgg16 import VGG16
from resnet164 import ResNet164
from mobilenet import MobileNet
from wide_resnet_28_10 import WideResNet28_10
import numpy as np
import os

PATH = './models/'
models = [VGG16(), ResNet164(), WideResNet28_10(), MobileNet()]
# models = [ResNet164(), WideResNet28_10()]
# models = [VGG16(), WideResNet28_10(), MobileNet()]
# models = [VGG16(), ResNet164(), MobileNet()]
# models = [VGG16(), ResNet164(), WideResNet28_10()] # 99.77% by majority
# models = [ResNet164(), WideResNet28_10(), MobileNet()] # Achieved 99.78% by Average Ensemble
# models = [VGG16(), MobileNet()]

def evaluate(prediction, true_label):
    '''
    Returns the test accuracy.

    Args:
        prediction - 2D numpy array (Number of samples, Class)
        y_test - 2D numpy array (Number of samples, Class)

    Returns:
        Test accuracy (float)
    '''
    pred_indices = np.argmax(prediction, 1)
    true_indices = np.argmax(true_label, 1)

    return np.mean(pred_indices == true_indices)

def unweighted_average_ensemble(predictions):
    '''
    Averge predictions of all models
    The following group acheives 99.78 % on test set.
    models = [ResNet164(), WideResNet28_10(), MobileNet()]

    Args:
        predictions - List of 2-D numpy arrays

    Retruns:
        final prediction - 2-D numpy array
    '''

    return np.mean(np.asarray(predictions), 0)

def majority_voting_ensemble(predictions):
    '''
    prediction by mojority voting
    The following groups acheiev 99.77 % on test set
    models = [VGG16(), ResNet164(), WideResNet28_10()]
    models = [ResNet164(), WideResNet28_10(), MobileNet()]

    Args:
        predictions - List of 2-D numpy arrays

    Retruns:
        final prediction - 2-D numpy array
    '''
    one_hot_predictions = []
    for prediction in predictions:
        one_hot_predictions.append(np_utils.to_categorical(np.argmax(prediction, axis = 1)))

    return np.sum(np.asarray(one_hot_predictions), 0)

def main():
    _, _, (x_test, y_test) = load_mnist()

    predictions = []
    for model in models:
        model_name = type(model).__name__
        model.compile()

        print('Loading pretrained weights for ', model_name, '...', sep='')
        model.load_weights(PATH + model_name + '.h5')

        # In order to save time, stored prediction results can be used.
        prediction_path = './predictions/' + model_name + '_test.npy'
        if os.path.isfile(prediction_path):
            single_model_prediction = np.load(prediction_path)
            print('Prediction file loaded')
        else:
            print('No prediction file. Predicting...')
            single_model_prediction = model.predict(x_test, verbose = 1)
            np.save(prediction_path, single_model_prediction)
            print('Saved prediction file in', prediction_path)

        predictions.append(single_model_prediction)
        single_model_accuracy = evaluate(single_model_prediction, y_test)
        print(f'Evaluation of {model_name}:', single_model_accuracy * 100, '%')
        print()

    ensemble_accuracy = evaluate(unweighted_average_ensemble(predictions), y_test)
    print('Accuracy by Unweighted Average Ensemble: ', ensemble_accuracy * 100, '%')

    ensemble_accuracy = evaluate(majority_voting_ensemble(predictions), y_test)
    print('Accuracy by Majority Voting Ensemble: ', ensemble_accuracy * 100, '%')

if __name__ == '__main__':
    main()
