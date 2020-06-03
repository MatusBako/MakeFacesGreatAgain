#!/usr/bin/env python3

from torchvision import models
from torchsummary import summary
from torch import device
from models.MFGAn import Generator
#from model import Net

device = 'cpu'
model = Generator(4).to(device)
print(summary(model, (3, 64, 64), device=device, batch_size=4))
