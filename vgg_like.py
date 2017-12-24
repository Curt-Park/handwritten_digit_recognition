from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers import (Input, Conv2D, BatchNormalization,
                          Activation, Dense, Dropout, Flatten)
from keras.models import Model
from model import BaseModel
import utils
import argparse

MODEL_NAME = 'VGGLike' # This should be modified when the model name changes.
WEIGHTS_PATH = f'./models/{MODEL_NAME}.h5'
IMAGE_PATH = f'./images/{MODEL_NAME}.png'
PLOT_PATH  = f'./images/{MODEL_NAME}_plot.png'

class VGGLike(BaseModel):
    '''
    A simple model with 2 FC layers.
    '_build()' is only modified when the model changes.

    HowToUse:
        model = VGGLike(PATH_FOR_WEIGHTS)
        * all funtionalities are written in BaseModel.py
    '''
    def __init__(self, path_for_weights):
        callbacks = [ModelCheckpoint(path_for_weights, save_best_only=True),
                     ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 5)]
        BaseModel.__init__(self, self._build(), callbacks)

    def _build(self):
        x = Input(shape = (28, 28, 1))
        y = self._3conv_block(x, out_channel = 32, name = 'block1')
        y = self._3conv_block(y, out_channel = 64, name = 'block2')
        y = self._channel_reduction_conv(y, out_channel = 64, name = 'channel_reduction1')
        y = self._channel_reduction_conv(y, out_channel = 32, name = 'channel_reduction2')
        y = self._channel_reduction_conv(y, out_channel = 16, name = 'channel_reduction3')
        y = self._channel_reduction_conv(y, out_channel =  8, name = 'channel_reduction4')
        y = self._channel_reduction_conv(y, out_channel =  4, name = 'channel_reduction5')
        y = Flatten()(y)
        y = self._dense_relu_dropout(y, out_units = 64, name = 'dense1')
        y = self._dense_relu_dropout(y, out_units = 32, name = 'dense2')
        y = Dense(units = 10, activation='softmax', name = 'dense_softmax')(y)
        return Model(x, y, name = MODEL_NAME)

    def _3conv_block(self, x, out_channel, name):
        y = Conv2D(out_channel,     (3, 3), padding = 'same', name = f'{name}_conv1')(x)
        y = Activation('relu')(BatchNormalization()(y))
        y = Conv2D(out_channel,     (3, 3), padding = 'same', name = f'{name}_conv2')(y)
        y = Activation('relu')(BatchNormalization()(y))
        y = Conv2D(out_channel * 2, (2, 2), strides = (2,2), padding = 'valid',
                                                             name = f'{name}_pooling_conv3')(y)
        y = Activation('relu')(BatchNormalization()(y))
        y = Dropout(rate = 0.1)(y)
        return y

    def _channel_reduction_conv(self, x, out_channel, name):
        y = Conv2D(out_channel, (1, 1), padding = 'same', name = name)(x)
        y = Activation('relu')(BatchNormalization()(y))
        return y

    def _dense_relu_dropout(self, x, out_units, name):
        y = Dense(out_units, name = name)(x)
        y = Activation('relu')(BatchNormalization()(y))
        y = Dropout(rate = 0.2)(y)
        return y

def get_argument_parser():
    '''
    Argument parser which returns the options which the user inputted.

    Args:
        None

    Returns:
        argparse.ArgumentParser().parse_args()
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',
                        help = 'How many epochs you need to run (default: 10)',
                        type = int, default = 10)
    parser.add_argument('--batch_size',
                        help = 'The number of images in a batch (default: 32)',
                        type = int, default = 32)
    parser.add_argument('--path_for_weights',
                        help = f'The path from where the weights will be saved or loaded \
                                (default: {WEIGHTS_PATH})',
                        type = str, default = WEIGHTS_PATH)
    parser.add_argument('--path_for_image',
                        help = f'The path from where the model image will be saved \
                                (default: {IMAGE_PATH})',
                        type = str, default = IMAGE_PATH)
    parser.add_argument('--path_for_plot',
                        help = f'The path from where the training progress will be plotted \
                                (default: {PLOT_PATH})',
                        type = str, default = PLOT_PATH)
    parser.add_argument('--data_augmentation',
                        help = '0: No, 1: Yes (default: 1)',
                        type = int, default = 1)
    parser.add_argument('--load_weights',
                        help = '0: No, 1: Yes (default: 0)',
                        type = int, default = 0)
    parser.add_argument('--plot_training_progress',
                        help = '0: No, 1: Yes (default: 1)',
                        type = int, default = 1)
    parser.add_argument('--save_model_to_image',
                        help = '0: No, 1: Yes (default: 1)',
                        type = int, default = 1)
    args = parser.parse_args()

    return args

def main():
    # load all arguments
    args = get_argument_parser()

    training_data, validation_data, test_data = utils.load_mnist()
    print(f'[data loaded]')

    # build and compile the model
    model = VGGLike(args.path_for_weights)
    model.compile()
    print('[model built]')

    # save the model architecture to an image file
    if args.save_model_to_image:
        model.save_model_as_image(args.path_for_image)
        print(f'[model image saved as {args.path_for_image}]')

    # load pretrained weights
    if args.load_weights:
        model.load_weights(args.path_for_weights)
        print(f'[weights loaded from {args.path_for_weights}]')

    # train the model
    hist = None
    if args.data_augmentation:
        hist = model.fit_generator(training_data, validation_data,
                                      epochs = args.epochs, batch_size = args.batch_size)
        print('[trained with augmented images]')
    else:
        hist = model.fit(training_data, validation_data,
                            epochs = args.epochs, batch_size = args.batch_size)
        print('[trained without augmented images]')

    # save the training progress to an image file
    if args.plot_training_progress:
        utils.plot(history = hist, path = args.path_for_plot, title = MODEL_NAME)
        print(f'training progress saved as {args.path_for_plot}')

    # evaluate the model with the test dataset
    loss_and_metrics = model.evaluate(test_data, batch_size = args.batch_size)
    print('Evaluation on the test dataset\n', loss_and_metrics, '\n')

if __name__ == '__main__':
    main()
