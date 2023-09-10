# Flower Breed Classifier

This project is an image classifier that can identify 102 different types of flowers. It uses pytorch, a popular deep learning framework, to build and train a neural network model based on one of the following architectures: densenet121, alexnet, or vgg16. The model can be trained on a dataset of flower images, saved as a checkpoint, and used to make predictions on new images.

## Dataset

The dataset used for this project is a subset of the [102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) created by Maria-Elena Nilsback and Andrew Zisserman. It contains 8189 images of flowers belonging to 102 categories. The images have various sizes and orientations, and some categories are more represented than others.

The dataset is divided into three folders: train, valid, and test. The train folder contains 6552 images for training the model. The valid folder contains 818 images for validating the model during training. The test folder contains 819 images for testing the model after training.

The dataset also includes a file called cat_to_name.json that maps each category code (from 1 to 102) to the corresponding flower name.

## Installation

To run this project, you need to have python 3 installed on your system. You also need to install the following packages:

- PIL
- matplotlib
- torch
- torchvision
- numpy
- json
- argparse

You can install these packages using pip or conda.

You also need to download the dataset from [here] and unzip it in your working directory.

## Usage

This project consists of four python files:

- data_manager.py: This file contains functions for loading and transforming the data, and mapping the category codes to names.
- model_manager.py: This file contains functions for creating, saving, and loading the model.
- train_engine.py: This file contains functions for training and validating the model.
- train.py: This file is the main script for training the model. It takes several arguments from the command line, such as data directory, save directory, model architecture, learning rate, hidden units, epochs, and gpu.
- predict.py: This file is the main script for predicting an image using the model. It takes several arguments from the command line, such as model name, image directory, checkpoint path, top k probabilities, gpu, category names, and hidden units.

To train the model, you can run the train.py script with the following command:

`python train.py data_directory --save_dir save_directory --arch model_architecture --learning_rate learning_rate --hidden_units hidden_units --epochs epochs --gpu`

For example:

`python train.py flowers --save_dir checkpoints --arch densenet121 --learning_rate 0.001 --hidden_units 512 --epochs 5 --gpu`

This will train a densenet121 model on the flowers dataset for 5 epochs with a learning rate of 0.001 and 512 hidden units in the classifier layer. It will also use gpu if available and save the model checkpoint in the checkpoints folder.

To predict an image using the model, you can run the predict.py script with the following command:

`python predict.py model_name image_directory checkpoint --top_k top_k --gpu --category_names category_names --hidden_units hidden_units`

For example:

`python predict.py densenet121 flowers/test/1/image_06743.jpg checkpoints/asdcheckpoint.pth --top_k 3 --gpu --category_names cat_to_name.json --hidden_units 512`

This will load a densenet121 model with 512 hidden units from the checkpoints folder and use it to predict the flower name and probability for the image flowers/test/1/image_06743.jpg. It will also use gpu if available and map the category codes to names using the cat_to_name.json file. It will display the top 3 probabilities and names for the image.
