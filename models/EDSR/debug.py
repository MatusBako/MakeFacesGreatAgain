#!/usr/bin/env python3

from torchvision import models
from torchsummary import summary
from torch import device
from models.EDSR.model import Net

model = Net(3, 4).to(device("cuda:0"))
print(summary(model, (3, 44, 54)))
