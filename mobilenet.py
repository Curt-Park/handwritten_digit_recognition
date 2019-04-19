from keras.callbacks import ReduceLROnPlateau
from keras.applications.mobilenet import DepthwiseConv2D
from keras.layers import (Input, Conv2D, BatchNormalization, ZeroPadding2D,
                          GlobalAveragePooling2D, Activation, Dense)
from keras.models import Model
from keras import optimizers
from base_model import BaseModel
from train import train

ALPHA = 1
MODEL_NAME = f'MobileNet' # This should be modified when the model name changes.

class MobileNet(BaseModel):
    '''
    1. ZeroPadding2D (2, 2)
    2. 3X3 Conv2D 32
    3. Depthwise separable convolution block X 13
    4. GlobalAveragePooling2D
    5. FC 10 + Softmax

    '_build()' is only modified when the model changes.

    HowToUse:
        model = MobileNet()
        * all funtionalities are written in BaseModel.py
    '''
    def __init__(self):
        '''
        - Reference for hyperparameters
          => https://github.com/Zehaos/MobileNet/issues/13
        '''
        callbacks = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1,
                                       patience = 30, verbose = 1)]
        optimizer = optimizers.RMSprop(lr = 0.01)
        BaseModel.__init__(self, model = self._build(), optimizer = optimizer,
                           callbacks = callbacks)

    def _build(self):
        '''
        Builds MobileNet.
        - MobileNets (https://arxiv.org/abs/1704.04861)
          => Depthwise Separable convolution
          => Width multiplier
        - Implementation in Keras
          => https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet.py
        - How Depthwise conv2D works
          => https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d

        Returns:
            MobileNet model
        '''
        alpha = ALPHA # 0 < alpha <= 1
        x = Input(shape = (28, 28, 1))
        y = ZeroPadding2D(padding = (2, 2))(x) # matching the image size of CIFAR-10

        # some layers have different strides from the papers considering the size of mnist
        y = Conv2D(int(32 * alpha), (3, 3), padding = 'same')(y) # strides = (2, 2) in the paper
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = self._depthwise_sep_conv(y, 64, alpha) # spatial size: 32 x 32
        y = self._depthwise_sep_conv(y, 128, alpha, strides = (2, 2)) # spatial size: 32 x 32
        y = self._depthwise_sep_conv(y, 128, alpha) # spatial size: 16 x 16
        y = self._depthwise_sep_conv(y, 256, alpha, strides = (2, 2)) # spatial size: 8 x 8
        y = self._depthwise_sep_conv(y, 256, alpha) # spatial size: 8 x 8
        y = self._depthwise_sep_conv(y, 512, alpha, strides = (2, 2)) # spatial size: 4 x 4
        for _ in range(5):
            y = self._depthwise_sep_conv(y, 512, alpha) # spatial size: 4 x 4
        y = self._depthwise_sep_conv(y, 1024, alpha, strides = (2, 2)) # spatial size: 2 x 2
        y = self._depthwise_sep_conv(y, 1024, alpha) # strides = (2, 2) in the paper
        y = GlobalAveragePooling2D()(y)
        y = Dense(units = 10)(y)
        y = Activation('softmax')(y)

        return Model(x, y, name = MODEL_NAME)

    def _depthwise_sep_conv(self, x, filters, alpha, strides = (1, 1)):
        '''
        Creates a depthwise separable convolution block

        Args:
            x - input
            filters - the number of output filters
            alpha - width multiplier
            strides - the stride length of the convolution

        Returns:
            A depthwise separable convolution block
        '''
        y = DepthwiseConv2D((3, 3), padding = 'same', strides = strides)(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv2D(int(filters * alpha), (1, 1), padding = 'same')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        return y

def main():
    '''
    Train the model defined above.
    '''
    model = MobileNet()
    train(model, MODEL_NAME)

if __name__ == '__main__':
    main()
