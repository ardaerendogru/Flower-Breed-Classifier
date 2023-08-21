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

parser = argparse.ArgumentParser(description = 'Predict an image using classifier model.')
parser.add_argument('model_name', help = 'Model for prediction.')
parser.add_argument('image_directory', help = 'Path for the image')
parser.add_argument('checkpoint', help = 'Path for the checkpoint.')
parser.add_argument('--top_k', help = 'Max k probabilities and names for the image.')
parser.add_argument('--gpu', help = 'Use GPU for training.', action='store_true')
parser.add_argument('--category_names', help = 'Category mapping with json file.')
parser.add_argument('--hidden_units', help = 'Number of hidden units for neural network.')


args = parser.parse_args()
model_name = args.model_name
image_directory = args.image_directory
checkpoint = args.checkpoint
top_k = 1 if args.top_k is None else int(args.top_k)
gpu = False if args.gpu is None else True
hidden_units = 512 if args.hidden_units is None else args.hidden_units
category_names = 'cat_to_name.json' if args.category_names is None else args.category_names

device = 'cuda' if torch.cuda.is_available() else 'cpu'

match model_name:
    case 'densenet121':
      weights = torchvision.models.DenseNet121_Weights.DEFAULT
      input_size = 1024
    case 'alexnet':
      weights = torchvision.models.AlexNet_Weights.DEFAULT
      input_size = 9216    
    case 'vgg16':
      weights = torchvision.models.VGG16_Weights.DEFAULT
      input_size = 25088
auto_transforms = weights.transforms()
model = model_manager.load_model(checkpoint, model_name, hidden_units)
img = Image.open(image_directory)
img = np.array(auto_transforms(img))
image = torch.from_numpy(img).type(torch.FloatTensor)
image = image.unsqueeze(0)
image = image.to(device)
model.to(device)
model.eval()
with torch.inference_mode():
  output = model(image)
output_prob = torch.softmax(output, dim=1)
probs, indeces = output_prob.topk(top_k)
probs   = probs.to('cpu').numpy().tolist()[0]
indeces = indeces.to('cpu').numpy().tolist()[0]
code_to_name = data_manager.code_to_name(category_names)
class_names = [code_to_name[f'{i+1}'] for i in indeces ]
data_dir = 'flowers'
model.class_to_idx = data_manager.get_class_to_idx(data_dir,model_name)

print(probs)
mapping = {val: key for key, val in model.class_to_idx.items()}
classes = [mapping[item] for item in indeces]
print(classes)
print(class_names)
