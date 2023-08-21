from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import json
from manager import data_manager, train_engine, model_manager
import argparse
import torch
import os
parser = argparse.ArgumentParser(description = 'Train an image classifier model for given dataset.')
parser.add_argument('data_directory', help = 'Path for the dataset')
parser.add_argument('--save_dir', help = 'Path for saving directory.')
parser.add_argument('--arch', help = 'Model architecture : {vgg16, densenet121, alexnet}')
parser.add_argument('--learning_rate', help = 'Learning rate for optimizer.')
parser.add_argument('--hidden_units', help = 'Number of hidden units for neural network.')
parser.add_argument('--epochs', help = 'Number of epochs')
parser.add_argument('--gpu', help = 'Use GPU for training.', action='store_true')

args = parser.parse_args()
data_dir = args.data_directory
save_dir = './arda' if args.save_dir is None else args.save_dir
model_name = 'densenet121' if args.arch is None else args.arch
lr = 0.001 if args.learning_rate is None else int(args.learning_rate)
hidden_units = 512 if args.hidden_units is None else float(args.hidden_units)
epochs = 5 if args.epochs is None else int(args.epochs)
gpu = False if args.gpu is None else True
train_loader, test_loader, valid_loader = data_manager.load_data(data_dir = data_dir, model_name = model_name, batch_size = 64)
model = model_manager.create_model(model_name, hidden_units)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = train_engine.train(model, optimizer, loss_fn, train_loader, valid_loader, epochs, device)
torch.save(model.state_dict(), os.path.join(save_dir, 'asdcheckpoint.pth'))
