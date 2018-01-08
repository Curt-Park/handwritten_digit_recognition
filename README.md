# Handwritten digit recognition with MNIST and Keras

This repository is for practice of implementing well-known network architectures and ensembling methods, including the followings:

#### Architectures
- [Mobilenet](https://arxiv.org/abs/1704.04861) - [[structure]](images/MobileNet.png) [[training progress]](images/MobileNet_9968_plot.png)
- [VGG16](https://arxiv.org/abs/1409.1556) - [[structure]](images/VGG16.png) [[training progress]](images/VGG16_9968_plot.png)
- [Resnet164](https://arxiv.org/abs/1603.05027) - [[structure]](images/ResNet164.png) [[training progress]](images/ResNet164_9970_plot.png)
- [WideResnet28-10](https://arxiv.org/abs/1603.05027) - [[structure]](images/WideResNet28_10.png) [[training progress]](images/WideResNet28_10_9976_plot.png)

#### Ensembling methods
- Unweighted average
- Majority voting
- [Super Learner](https://arxiv.org/abs/1704.01664) - [[structure]](images/SuperLearner.png)

#### Others
- Channel-wise normalization of input images: substracted by mean and divided by std
- Data augmentation: rotation, width shift, height shift, shearing, zooming

## Environment
- MacOS High Sierra 10.13.1 for implementation / Ubuntu 14.04 for training
- Python 3.6.3
- Keras 2.1.2 (Tensorflow backend)

## Evaluation
The best single model and the best ensemble method achieve **99.76%** and **99.77%** on the test set respectively.

|  **Model**          |  **On the validation set**  | **On the test set** |
|:-------------------:|:---------------------------:|:-------------------:|
|  Mobilenet          |          99.63%             |       99.68%        |
|  VGG16              |          99.61%             |       99.68%        |
|  Resnet164          |        **99.72%**           |       99.70%        |
|  WideResnet28-10    |        **99.72%**           |     **99.76%**      |

|  **Ensemble (all)** |  **On the validation set**  | **On the test set** |
|:-------------------:|:---------------------------:|:-------------------:|
|  Unweighted average |          99.70%             |       99.75%        |
|  Majority voting    |          99.71%             |       99.76%        |
|  Super Learner      |        **99.73%**           |     **99.77%**      |

In order to run the evaluation, it requires pre-trained weights for each model, which can be downloaded [here](https://drive.google.com/drive/folders/1kBvOL019Gcx00vwUM1BhtaoWQDlfPPim?usp=sharing).

**\*All pre-trained weights should be stored in './models'.**

#### How to run
```bash
python evaluate.py [options]
```
#### Options
```bash
$ python evaluate.py --help
usage: evaluate.py [-h] [--dataset DATASET]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  training set: 0, validation set: 1, test set: 2
```

## Training
The training can be executed by the following command. Every model has the same options.

#### How to run
```bash
$ python vgg16.py [options]
```

#### Options
```bash
$ python vgg16.py --help
usage: vgg16.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                [--path_for_weights PATH_FOR_WEIGHTS]
                [--path_for_image PATH_FOR_IMAGE]
                [--path_for_plot PATH_FOR_PLOT]
                [--data_augmentation DATA_AUGMENTATION]
                [--save_model_and_weights SAVE_MODEL_AND_WEIGHTS]
                [--load_weights LOAD_WEIGHTS]
                [--plot_training_progress PLOT_TRAINING_PROGRESS]
                [--save_model_to_image SAVE_MODEL_TO_IMAGE]

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       How many epochs you need to run (default: 10)
  --batch_size BATCH_SIZE
                        The number of images in a batch (default: 64)
  --path_for_weights PATH_FOR_WEIGHTS
                        The path from where the weights will be saved or
                        loaded (default: ./models/VGG16.h5)
  --path_for_image PATH_FOR_IMAGE
                        The path from where the model image will be saved
                        (default: ./images/VGG16.png)
  --path_for_plot PATH_FOR_PLOT
                        The path from where the training progress will be
                        plotted (default: ./images/VGG16_plot.png)
  --data_augmentation DATA_AUGMENTATION
                        0: No, 1: Yes (default: 1)
  --save_model_and_weights SAVE_MODEL_AND_WEIGHTS
                        0: No, 1: Yes (default: 1)
  --load_weights LOAD_WEIGHTS
                        0: No, 1: Yes (default: 0)
  --plot_training_progress PLOT_TRAINING_PROGRESS
                        0: No, 1: Yes (default: 1)
  --save_model_to_image SAVE_MODEL_TO_IMAGE
                        0: No, 1: Yes (default: 1)
```

## File descriptions
```bash
├── images/ # model architectures and training progresses
├── predictions/ # prediction results to be used for fast inference
├── models/ # model weights (not included in this repo)
├── README.md
├── base_model.py # base model interface
├── evaluate.py # for evaluation
├── utils.py # helper functions
├── mobilenet.py
├── vgg16.py
├── resnet164.py
├── wide_resnet_28_10.py
└── super_learner.py
```

## References
#### Papers
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
- [Wide Residual Networks](https://arxiv.org/abs/1605.07146)
- [The Relative Performance of Ensemble Methods with Deep Convolutional Neural Networks for Image Classification](https://arxiv.org/abs/1704.01664)

#### Implementation
- [ResNet Author's Implementation](https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua)
- [WideResNet Author's implementation](https://github.com/szagoruyko/wide-residual-networks)
- [MobileNet in Keras](https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet.py)
- [How far can we go with MNIST??](https://github.com/hwalsuklee/how-far-can-we-go-with-MNIST)

#### Others
- [Global weight decay in keras? - Stackoverflow](https://stackoverflow.com/questions/41260042/global-weight-decay-in-keras)
- [Best up to date result on MNIST dataset - Kaggle](https://www.kaggle.com/c/digit-recognizer/discussion/23999#138390)
- [Batch Normalization before or after ReLU? - Reddit](https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/)
- [Depth-wise Conv2D - Tensorflow Document](https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d)
- [A Complete Tutorial on Ridge and Lasso Regression in Python](https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/)
