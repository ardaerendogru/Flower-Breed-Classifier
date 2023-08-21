from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import json

def load_data(data_dir,
              model_name,
              batch_size):
  
  train_dir = data_dir + '/train'
  valid_dir = data_dir + '/valid'
  test_dir = data_dir + '/test'


  
  match model_name:
    case 'densenet121':
      weights = torchvision.models.DenseNet121_Weights.DEFAULT
    case 'alexnet':
      weights = torchvision.models.AlexNet_Weights.DEFAULT
    case 'vgg16':
      weights = torchvision.models.VGG16_Weights.DEFAULT
  auto_transforms = weights.transforms()
  train_dataset = datasets.ImageFolder(train_dir, transform = auto_transforms)
  valid_dataset = datasets.ImageFolder(valid_dir, transform = auto_transforms)
  test_dataset = datasets.ImageFolder(test_dir, transform = auto_transforms)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
  valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size)
  return train_loader, test_loader, valid_loader

def get_class_to_idx(data_dir,
              model_name):
  
  train_dir = data_dir + '/train'
  valid_dir = data_dir + '/valid'
  test_dir = data_dir + '/test'


  
  match model_name:
    case 'densenet121':
      weights = torchvision.models.DenseNet121_Weights.DEFAULT
    case 'alexnet':
      weights = torchvision.models.AlexNet_Weights.DEFAULT
    case 'vgg16':
      weights = torchvision.models.VGG16_Weights.DEFAULT
  auto_transforms = weights.transforms()
  train_dataset = datasets.ImageFolder(train_dir, transform = auto_transforms)
  return train_dataset.class_to_idx
def code_to_name(path:str):
  with open(path, 'r') as f:
    cat_to_name = json.load(f)
  return cat_to_name
