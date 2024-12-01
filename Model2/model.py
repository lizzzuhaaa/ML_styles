from DataPrepare.DataManager import *
import torch.nn as nn
from torchvision import models


# Using Pre-trained model ResNet-50
model = models.resnet50(pretrained=True)

# Modifying params for a specified training
for param in model.parameters():
    param.requires_grad = False

# Modifying last layer for the classification
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(classes))

# Check model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# print(model)
