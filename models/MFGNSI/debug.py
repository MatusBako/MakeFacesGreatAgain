#!/usr/bin/env python3

from torchsummary import summary
from torchvision import models
from torch import device
from models.MFGNSI import Net


model = Net(4).to(device("cuda:0"))
print(summary(model, (3, 256, 256)))
