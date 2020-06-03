#!/usr/bin/env python3

from torchvision import models
from torchsummary import summary
from torch import device
from models.RCAN import Net
#from model import Net

device = 'cpu'
model = Net(4).to(device)
print(summary(model, (3, 64, 64), device=device, batch_size=8))