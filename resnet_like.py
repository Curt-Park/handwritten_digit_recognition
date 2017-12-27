from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers import (Input, Conv2D, BatchNormalization, ZeroPadding2D,
                          AveragePooling2D, Activation, Dense, Flatten, add)
from keras.models import Model
from keras import optimizers
from model import BaseModel
import utils
import argparse

MODEL_NAME = 'ResNetLike' # This should be modified when the model name changes.
WEIGHTS_PATH = f'./models/{MODEL_NAME}.h5'
IMAGE_PATH = f'./images/{MODEL_NAME}.png'
PLOT_PATH  = f'./images/{MODEL_NAME}_plot.png'

class ResNetLike(BaseModel):
    '''
    1. 3X3 Conv2D 32 + BN + Relu
    2. 3X3 Conv2D 32 + BN + Relu
    3. 3X3 Conv2D 64 with strides (2, 2) + Dropout 0.1
    4. 3X3 Conv2D 64 + BN + Relu
    5. 3X3 Conv2D 64 + BN + Relu
    6. 3X3 Conv2D 128 with strides (2, 2) + Dropout 0.1
    7. 1X1 Conv2D 64 + BN + Relu
    8. 1X1 Conv2D 32 + BN + Relu
    9. 1X1 Conv2D 16 + BN + Relu
    10. 1X1 Conv2D 8 + BN + Relu
    11. 1X1 Conv2D 4 + BN + Relu
    12. FC 64 + BN + Relu + Dropout 0.2
    13. FC 32 + BN + Relu + Dropout 0.2
    14. FC 10 + Softmax

    '_build()' is only modified when the model changes.

    HowToUse:
        model = VGGLike(PATH_FOR_WEIGHTS)
        * all funtionalities are written in BaseModel.py
    '''
    def __init__(self, path_for_weights):
        callbacks = [ModelCheckpoint(path_for_weights, save_best_only=True, verbose = 1),
                     ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1,
                                       patience = 10, verbose = 1)]
        optimizer = optimizers.SGD(lr=0.1, momentum=0.9, decay=1e-04)
        BaseModel.__init__(self, model = self._build(), optimizer = optimizer,
                           callbacks = callbacks)

    def _build(self):
        # block_info: list of sets (dimensional_change, layer_num, inner_channel, out_channel)
        block_info = [(False, 3, 32, 64), (True, 3, 64, 128), (True, 5, 64, 256)]

        x = Input(shape = (28, 28, 1))
        y = ZeroPadding2D(padding = (2, 2))(x) # matching the size to CIFAR-10

        y = Conv2D(64, (7, 7), strides = (2,2), padding = 'same', name = '7x7_conv')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        # No MaxPooling2D

        i = 1
        for dimensional_change, layer_num, inner, out in block_info:
            if dimensional_change:
                y = self._residual_block_dimensional_change(y, out_channel = out,
                                                            name = f'block{i}')
                i += 1

            for j in range(layer_num):
                y = self._residual_block(y, inner_channel = inner, out_channel = out,
                                         name = f'block{i}')
                i += 1

        y = AveragePooling2D(strides = (2, 2))(y)

        y = Flatten()(y)
        y = Dense(units = 10, activation='softmax', name = 'dense_softmax')(y)

        return Model(x, y, name = MODEL_NAME)

    def _residual_block(self, x, inner_channel, out_channel, name):
        '''
        - Deep Residual Learning for Image Recognitio (https://arxiv.org/abs/1512.03385)
          : Bottleneck Architecture
        - Identity Mappings in Deep Residual Networks (https://arxiv.org/abs/1603.05027)
          : Full pre-activation
        '''
        fx = BatchNormalization()(x)
        fx = Activation('relu')(fx)
        fx = Conv2D(inner_channel, (1, 1), padding = 'same', kernel_initializer = 'he_normal',
                    name = f'{name}_conv1', )(fx)

        fx = BatchNormalization()(fx)
        fx = Activation('relu')(fx)
        fx = Conv2D(inner_channel, (3, 3), padding = 'same', kernel_initializer = 'he_normal',
                    name = f'{name}_conv2', )(fx)

        fx = BatchNormalization()(fx)
        fx = Activation('relu')(fx)
        fx = Conv2D(out_channel, (1, 1), padding = 'same', kernel_initializer = 'he_normal',
                    name = f'{name}_conv3', )(fx)

        hx = x

        return add([hx, fx])

    def _residual_block_dimensional_change(self, x, out_channel, name):
        '''
        It reduces the size of the input with regards to H and W
        and increases the channel number.

        - Deep Residual Learning for Image Recognitio (https://arxiv.org/abs/1512.03385)
          : Projection shorcut (B)
        - Identity Mappings in Deep Residual Networks (https://arxiv.org/abs/1603.05027)
          : Full pre-activation
        '''
        # No bottleneck architecture when the dimensionality changes.
        fx = BatchNormalization()(x)
        fx = Activation('relu')(fx)
        fx = Conv2D(out_channel, (3, 3), padding = 'same', strides = (2, 2),
                    kernel_initializer = 'he_normal', name = f'{name}_conv1', )(fx)

        fx = BatchNormalization()(fx)
        fx = Activation('relu')(fx)
        fx = Conv2D(out_channel, (3, 3), padding = 'same', kernel_initializer = 'he_normal',
                    name = f'{name}_conv2', )(fx)

        # Projection shorcut
        hx = Conv2D(out_channel, (1, 1), strides = (2, 2), kernel_initializer = 'he_normal',
                    name = f'{name}_projection_shorcut', )(x)

        return add([hx, fx])

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
