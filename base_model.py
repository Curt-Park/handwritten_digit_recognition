from keras.utils import plot_model
import utils

class BaseModel(object):
    def __init__(self, model, optimizer, callbacks = None):
        self.model = model
        self.callbacks = callbacks
        self.optimizer = optimizer

    def load_weights(self, path):
        self.model.load_weights(path)

    def save(self, path):
        self.model.save(path)

    def compile(self):
        self.model.compile(optimizer = self.optimizer, loss = 'categorical_crossentropy',
                         metrics = ['accuracy'])

    def fit(self, training_data, validation_data, epochs, batch_size):
        x_train, y_train = training_data
        x_val, y_val = validation_data

        hist = self.model.fit(x_train, y_train, epochs = epochs,
                              batch_size = batch_size,
                              validation_data = (x_val, y_val), callbacks = self.callbacks)
        return hist

    def fit_generator(self, training_data, validation_data, epochs, batch_size):
        x_train, y_train = training_data
        x_val, y_val = validation_data

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

    def evaluate(self, eval_data, batch_size = 32):
        x, y = eval_data
        loss_and_metrics = self.model.evaluate(x, y,
                                               batch_size = batch_size)
        return loss_and_metrics

    def predict(self, x, batch_size = None, verbose = 1, steps = None):
        return self.model.predict(x, batch_size, verbose, steps)

    def save_model_as_image(self, path):
        plot_model(self.model, to_file = path, show_shapes = True)
