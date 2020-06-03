#!/usr/bin/env python3

from torchvision import models
from torchsummary import summary
from torch import device
from models.EDSR.model import Net

model = Net(2, 3, base_channel=128, num_residuals=16).to(device("cuda:0"))
print(summary(model, (3, 128, 128), batch_size=8))
