from keras.models import Sequential
from keras.layers import Dense, Flatten
from model import BaseModel
import utils
import argparse

MODEL_NAME = 'ExampleModel' # This should be modified when the model name changes.
MODEL_PATH = f'./models/{MODEL_NAME}.h5'
IMAGE_PATH = f'./images/{MODEL_NAME}.png'

class ExampleModel(BaseModel):
    '''
    A simple model with 2 FC layers.
    '_build()' is only modified when the architecture changes.

    HowToUse:
        model = ExampleModel(PATH_TO_SAVE_OR_LOAD)
        * All funtionalities are written in BaseModel.py
    '''
    def __init__(self, path):
        BaseModel.__init__(self, self._build(), path)

    def _build(self):
        model = Sequential()
        model.add(Flatten(input_shape = (28, 28, 1)))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=10, activation='softmax'))

        return model

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
    parser.add_argument('--path',
                        help = f'The path from where the model will be saved or loaded \
                                (default: {MODEL_PATH})',
                        type = str, default = MODEL_PATH)
    parser.add_argument('--data_augmentation',
                        help = '0: No, 1: Yes (default: 1)',
                        type = int, default = 1)
    parser.add_argument('--load_weight',
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
    epochs = args.epochs
    batch_size = args.batch_size
    path = args.path
    data_augmentation = True if args.data_augmentation == 1 else False
    load_weight = False if args.load_weight == 0 else True
    plot_training_progress = True if args.plot_training_progress == 1 else False
    save_model_to_image = True if args.save_model_to_image == 1 else False

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = utils.load_mnist()
    print('[images loaded]')

    # build a model
    model = ExampleModel(path)
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    print('[model built]')

    if save_model_to_image:
        model.save_graphical_model(IMAGE_PATH)
        print('[model image saved]')

    if load_weight:
        model.load_weights()
        print('[weights loaded]')

    hist = None
    if data_augmentation:
        train_datagen = utils.get_train_generator(x_train, y_train, batch_size = batch_size)
        val_datagen = utils.get_val_generator(x_val, y_val, batch_size = batch_size)

        hist = model.fit_generator(train_datagen,
                                   steps_per_epoch = x_train.shape[0] // batch_size,
                                   epochs = epochs, validation_data = val_datagen,
                                   validation_steps = x_val.shape[0] // batch_size)
        print('[trained with augmented images]')

    else:
        hist = model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size,
                         validation_data = (x_val, y_val))
        print('[trained without augmented images]')

    loss_and_metrics = model.evaluate(x_test, y_test, batch_size = batch_size)
    print('Evaluation on the test dataset\n', loss_and_metrics, '\n')

    if plot_training_progress:
        model.plot(hist, MODEL_NAME)

if __name__ == '__main__':
    main()
