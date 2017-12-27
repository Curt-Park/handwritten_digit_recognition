from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers import (Input, Conv2D, BatchNormalization, ZeroPadding2D,
                          GlobalAveragePooling2D, Activation, Dense, add)
from keras.models import Model
from keras import optimizers
from model import BaseModel
import utils
import argparse

MODEL_NAME = 'ResNetLike' # This should be modified when the model name changes.
DEPTH = 164 # or 1001
WEIGHTS_PATH = f'./models/{MODEL_NAME}.h5'
IMAGE_PATH = f'./images/{MODEL_NAME}.png'
PLOT_PATH  = f'./images/{MODEL_NAME}_plot.png'

class ResNetLike(BaseModel):
    '''
    1. ZeroPadding2D (2, 2)
    2. 3X3 Conv2D 16
    3. ResidualBlock X 18 + 1
    4. ResidualBlock X 18 + 1
    5. ResidualBlock X 18 + 1
    6. BN + Relu
    7. GlobalAveragePooling2D
    8. FC 10 + Softmax

    '_build()' is only modified when the model changes.

    HowToUse:
        model = ResNetLike(PATH_FOR_WEIGHTS)
        * all funtionalities are written in BaseModel.py
    '''
    def __init__(self, path_for_weights):
        callbacks = [ModelCheckpoint(path_for_weights, save_best_only=True, verbose = 1),
                     ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1,
                                       patience = 5, verbose = 1)]
        optimizer = optimizers.SGD(lr=0.1, momentum=0.9, decay=1e-04)
        BaseModel.__init__(self, model = self._build(), optimizer = optimizer,
                           callbacks = callbacks)

    def _build(self):
        n = (DEPTH - 2) // 9
        nStages = [16, 64, 128, 256]

        x = Input(shape = (28, 28, 1))
        y = ZeroPadding2D(padding = (2, 2))(x) # matching the size as CIFAR-10

        y = Conv2D(nStages[0], (3, 3), padding = 'same')(y)
        y = self._layer(y, nStages[1], n, (1, 1)) # spatial size: 32 x 32
        y = self._layer(y, nStages[2], n, (2, 2)) # spatial size: 16 x 16
        y = self._layer(y, nStages[3], n, (2, 2)) # spatial size: 8 x 8
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = GlobalAveragePooling2D()(y)
        y = Dense(units = 10, activation='softmax')(y)

        return Model(x, y, name = MODEL_NAME)

    def _layer(self, x, output_channel, count, strides):
        '''
        Creates a layer which consists of residual blocks as many as 'count'.
        '''
        y = self._residual_block(x, output_channel, True, strides)

        for i in range(1, count):
            y = self._residual_block(y, output_channel, False, (1, 1))

        return y

    def _residual_block(self, x, output_channel, dimensional_increase, strides):
        '''
        It reduces the size of the input with regards to H and W
        and increases the channel number.

        - Deep Residual Learning for Image Recognitio (https://arxiv.org/abs/1512.03385)
          : Bottleneck
          : Projection shortcut (B)
        - Identity Mappings in Deep Residual Networks (https://arxiv.org/abs/1603.05027)
          : Full pre-activation
          : https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua
        '''
        bottleneck_channel = output_channel // 4

        if dimensional_increase:
            # common BN, ReLU
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

        # conv1x1
        fx = BatchNormalization()(x)
        fx = Activation('relu')(fx)
        fx = Conv2D(bottleneck_channel, (1, 1), padding = 'same', strides = strides,
                    kernel_initializer = 'he_normal')(fx)

        # conv3x3
        fx = BatchNormalization()(fx)
        fx = Activation('relu')(fx)
        fx = Conv2D(bottleneck_channel, (3, 3), padding = 'same',
                    kernel_initializer = 'he_normal')(fx)

        # conv1x1
        fx = BatchNormalization()(fx)
        fx = Activation('relu')(fx)
        fx = Conv2D(output_channel, (1, 1), padding = 'same',
                    kernel_initializer = 'he_normal')(fx)

        if dimensional_increase:
            # Projection shorcut
            x = Conv2D(output_channel, (1, 1), padding = 'same', strides = strides,
                        kernel_initializer = 'he_normal')(x)

        return add([x, fx])

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
                        type = int, default = 64)
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
    model = ResNetLike(args.path_for_weights)
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
