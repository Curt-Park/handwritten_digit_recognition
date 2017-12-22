import os
import matplotlib.pyplot as plt
from collections import defaultdict
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

class BaseModel(object):
    def __init__(self, model, path):
        '''
        Constructor

        Args:
            model: a model built by Keras
        '''
        self.model = model
        self.path = path

    def compile(self, optimizer, loss, metrics, **kwargs):
        '''
        Compiles the model

        Args:
            optimizer: string or any optimizer in keras.optimizers
            loss: string or any loss function in keras.losses
            metrics: list of metrics (ex: ['accuracy'])
            ...
            see: https://keras.io/models/sequential/#compile

        Returns:
            None
        '''
        self.model.compile(optimizer = optimizer, loss = loss, metrics = metrics, **kwargs)

    def fit(self, x, y, epochs = 10, batch_size = 32, validation_data = None, **kwargs):
        '''
        Trains the model

        Args:
            x: training input data (4-D array)
            y: training label data (2-D array)
            epochs: integer
            batch_size: integer
            validation_data: (x_val, y_val)
            ...
            see: https://keras.io/models/sequential/#fit

        Returns:
            History object: its History.history attribute is a record of training loss values
                            and metrics values at successive epochs, as well as
                            validation loss values and validation metrics values.
                            e.g. history['loss'] / history['val_loss'] /
                                 history['acc'] / history['val_acc]
        '''
        best_only = ModelCheckpoint(self.path, save_best_only=True)
        return self.model.fit(x, y, epochs = epochs, batch_size = batch_size,
                              validation_data = validation_data,
                              callbacks = [best_only], **kwargs)

    def fit_generator(self, generator, steps_per_epoch, epochs,
                      validation_data = None, validation_steps = None, **kwargs):
        '''
        Trains the model using genrator

        Args:
            generator: A generator. The output of the generator must be either.
            steps_per_epoch: integer
            epochs: integer
            validation_data: A generator for validation data
            validation_step: integer
            ...
            see: https://keras.io/models/sequential/#fit_generator

        Returns:
            History object: its History.history attribute is a record of training loss values
                            and metrics values at successive epochs, as well as
                            validation loss values and validation metrics values.
                            e.g. history['loss'] / history['val_loss'] /
                                 history['acc'] / history['val_acc]
        '''
        best_only = ModelCheckpoint(self.path, save_best_only=True)
        return self.model.fit_generator(generator, steps_per_epoch = steps_per_epoch,
                                        epochs = epochs, validation_data = validation_data,
                                        validation_steps = validation_steps,
                                        callbacks = [best_only], **kwargs)

    def evaluate(self, x, y, batch_size = 32, **kwargs):
        '''
        Evaluates the model

        Args:
            x: test input data (4-D array)
            y: test label data (2-D array)
            batch_size: integer
            ...
            see: https://keras.io/models/sequential/#evaluate

        Returns:
            loss_and_metrics: list
        '''
        return self.model.evaluate(x, y, batch_size = batch_size, **kwargs)

    def evaluate_generator(self, generator, steps = None, **kwargs):
        '''
        Evaluates the model on a data generator

        Args:
            generator: Generator yielding tuples
            steps: Total number of steps
            ...

        Returns:
            loss_and_metrics: list
        '''
        return self.model.evaluate_generator(generator, steps = steps, **kwargs)

    def predict(self, x, batch_size = 32, verbose = 0):
        '''
        Generates output predictions for the input samples

        Args:
            x: 4-D array
            batch_size = integer
            verbose = integer

        Returns:
            A numpy array of predictions
        '''
        return self.model.predict(x, batch_size = batch_size, verbose = verbose)

    def predict_generator(self, generator, steps = None, **kwargs):
        '''
        Generates output predictions from a data generator

        Args:
            generator: generator yielding batches
            steps: total number of steps
            ...
            see: https://keras.io/models/sequential/#evaluate_generator

        Returns:
            A numpy array of predictions
        '''
        return self.model.predict_generator(generator, steps = steps, **kwargs)

    def plot(self, history, title = None):
        '''
        Plot the trends of loss and metrics during training

        Args:
            history: History.history attribute. It is a return value of fit method.
            title: string

        Returns:
            None
        '''
        dhist = defaultdict(lambda: None) # just in case history doesn't have validation info
        dhist.update(history.history)

        _, loss_ax = plt.subplots()
        acc_ax = loss_ax.twinx()

        loss_ax.plot(dhist['loss'], 'y', label='training loss')
        if dhist['val_loss']:
            loss_ax.plot(dhist['val_loss'], 'r', label='validation loss')

        acc_ax.plot(dhist['acc'], 'b', label='training acc')
        if dhist['val_acc']:
            acc_ax.plot(dhist['val_acc'], 'g', label='validation acc')

        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        acc_ax.set_ylabel('accuracy')

        loss_ax.legend(loc='upper left')
        acc_ax.legend(loc='lower left')

        if title:
            plt.title(title)
        plt.show()

    def save_weights(self):
        '''
        Save weights
        It makes an error if their is no directory to save the model.

        Args:
            None

        Returns:
            None

        see: https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
        '''
        self.model.save_weights(self.path)

    def load_weights(self):
        '''
        Load a stored model and weights

        Args:
            None

        Returns:
            Keras model

        see: https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
        '''
        if os.path.isfile(self.path):
            self.model.load_weights(self.path)
        else:
            print("File Not Found")

    def save_graphical_model(self, path):
        '''
        Save an image file of a graph showing how the model looks like

        Args:
            path: string

        Returns:
            None
        '''
        plot_model(self.model, to_file = path, show_shapes = True)
