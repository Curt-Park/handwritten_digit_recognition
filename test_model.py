from keras.models import Sequential
from keras.layers import Dense, Flatten
from model import BaseModel
import numpy as np
import utils

class TestModel(BaseModel):
    def __init__(self, path):
        BaseModel.__init__(self, self._build(), path)

    def _build(self):
        model = Sequential()
        model.add(Flatten(input_shape = (28, 28, 1)))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=10, activation='softmax'))

        return model

# TODO: argument parser

def main():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = utils.load_mnist()
    batch_size = 32
    epochs = 10

    train_datagen = utils.get_train_generator(x_train, y_train)
    val_datagen = utils.get_val_generator(x_val, y_val)
    test_datagen = utils.get_test_generator(x_test, y_test)

    model = TestModel('./models/TestModel.h5')
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.fit_generator(train_datagen, steps_per_epoch = x_train.shape[0] // batch_size,
                        epochs = epochs, validation_data = val_datagen,
                        validation_steps = x_val.shape[0] // batch_size)
    loss_and_metrics = model.evaluate_generator(test_datagen)
    print('Evaluation\n', loss_and_metrics, '\n')

    output = model.predict_generator(test_datagen)
    print(np.sum(np.argmax(output, axis=1) == np.argmax(y_test, axis=1)))

if __name__ == '__main__':
    main()
