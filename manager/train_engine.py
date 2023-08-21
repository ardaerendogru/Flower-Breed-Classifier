from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import json
def train(model,optimizer , loss_fn, train_loader, valid_loader, epochs, device):
  
  for epoch in range(epochs):
    model.to(device)
    model.train()
    train_loss, train_acc = 0, 0
    
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
    train_loss = train_loss/len(train_loader)
    train_acc = train_acc/len(train_loader)
    
    model.eval()
    valid_loss, valid_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(valid_loader):
            X, y = X.to(device), y.to(device)
    
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            valid_loss += loss.item()
            test_pred_labels = torch.argmax(torch.softmax(test_pred_logits, dim=1), dim=1)
            valid_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))         
    valid_loss = valid_loss / len(valid_loader)
    valid_acc = valid_acc / len(valid_loader)
    print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"valid_loss: {valid_loss:.4f} | "
          f"valid_acc: {valid_acc:.4f}"
      )
  return model
