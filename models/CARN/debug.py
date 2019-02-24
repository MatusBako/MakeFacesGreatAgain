#!/usr/bin/env python3

from torchvision import models
from torchsummary import summary
from torch import device
from models.CARN import Net
#from model import Net


model = Net(2).to(device("cuda:0"))
print(summary(model, (3, 88, 108)))
