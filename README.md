# Handwritten digit recognition with MNIST and Keras

This repository is for practice of implementing well-known network architectures and ensembling methods, including the followings:

#### Architectures
- Mobilenet
- VGG16
- Resnet164
- WideResnet28-10

#### Ensembling methods
- Unweighted average
- Majority voting
- Super Learner

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
|  **Ensemble**       |  **On the validation set**  | **On the test set** |
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
