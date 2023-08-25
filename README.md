# AI Programming with Python Project

## Overview
This project is an image classifier that can identify 102 different breeds of flowers. The classifier is trained on a dataset of flower images, with each breed having between 40 and 258 images.

## Usage
To use this image classifier, you will need to provide the path to the dataset using the data_directory argument. You can also specify the following optional arguments:

--save_dir: Path for saving directory.
--arch: Model architecture (vgg16, densenet121, or alexnet).
--learning_rate: Learning rate for optimizer.
--hidden_units: Number of hidden units for neural network.
--epochs: Number of epochs.
--gpu: Use GPU for training.

Here is an example of how to use these arguments:
```
python train.py data_directory --save_dir save_directory --arch vgg16 --learning_rate 0.001 --hidden_units 512 --epochs 5 --gpu
```

This will train the model using the vgg16 architecture with a learning rate of 0.001, 512 hidden units, and 5 epochs. The model will be trained using a GPU and the results will be saved in the specified directory.
