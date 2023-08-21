from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import json
from torchvision import datasets, transforms, models
import torchvision
def create_model(model_name:str, hidden_units:int):
  match model_name:
    case 'densenet121':
      weights = torchvision.models.DenseNet121_Weights.DEFAULT
      model = torchvision.models.densenet121(weights = weights)
      input_size = 1024
    case 'alexnet':
      weights = torchvision.models.AlexNet_Weights.DEFAULT
      model = torchvision.models.AlexNet(weights = weights)
      input_size = 9216    
    case 'vgg16':
      weights = torchvision.models.VGG16_Weights.DEFAULT
      model = torchvision.models.vgg16(weights=weights)
      input_size = 25088
  for param in model.parameters():
    param.requires_grad = False
  
  model.classifier = model.classifier = nn.Sequential(
    nn.Linear(input_size,hidden_units),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(hidden_units,hidden_units),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(hidden_units,102)
)
  return model

def save_model(PATH:str, model:torchvision.models):
  torch.save(model.state_dict(), PATH)

def load_model(PATH: str,model_name:str, hidden_units:int):
  match model_name:
    case 'densenet121':
      weights = torchvision.models.DenseNet121_Weights.DEFAULT
      model = torchvision.models.densenet121(weights = weights)
      input_size = 1024
    case 'alexnet':
      weights = torchvision.models.AlexNet_Weights.DEFAULT
      model = torchvision.models.AlexNet(weights = weights)
      input_size = 9216    
    case 'vgg16':
      weights = torchvision.models.VGG16_Weights.DEFAULT
      model = torchvision.models.vgg16(weights=weights)
      input_size = 25088
  model.classifier = nn.Sequential(
    nn.Linear(input_size,hidden_units),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(hidden_units,hidden_units),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(hidden_units,102)
)
  model.load_state_dict(torch.load(PATH))
  return model
