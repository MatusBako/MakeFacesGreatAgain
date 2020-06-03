"""
Implementation of DRCN architecture. The implementation is inspired by implementation
from cited repository, but there were few changes made so it is usable with this module.


Core functionality inspired by: https://github.com/togheppi/pytorch-super-resolution-model-collection/blob/master/drcn.py
"""

from torch import nn, cat, ones, sum, tensor, zeros


class Net(nn.Module):
    def __init__(self, upscale_factor, base_channel=64):
        # base_channel = 256

        super(Net, self).__init__()
        self.upscale_factor = upscale_factor
        self.num_recursions = 8 #16
        input_channel_cnt = 3

        # used for loss computation
        self.reconstructed = []

        # embedding layer
        self.embedding_layer = nn.Sequential(
            nn.Conv2d(input_channel_cnt, base_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channel, base_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # inference layer
        self.conv_block = nn.Sequential(nn.Conv2d(base_channel, base_channel, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True))

        # reconstruction layer
        self.reconstruction_layer = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(base_channel, input_channel_cnt, kernel_size=3, stride=1, padding=1)
        )

        self.register_parameter("weight", nn.Parameter(ones(self.num_recursions) / self.num_recursions, requires_grad=True))
        self.weight_init()

    def forward(self, x: tensor):
        x = nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode="bicubic", align_corners=True)

        # embedding layer
        init_embedding = self.embedding_layer(x)

        # recursions
        infered = [init_embedding]
        for idx in range(self.num_recursions):
            infered.append(self.conv_block(infered[idx]))

        reconstructed = []
        out_sum = 0

        # iterate all inference levels
        for idx in range(self.num_recursions):
            reconstructed.append(self.reconstruction_layer(infered[idx + 1]))

            # each reconstruction is weighted and added to result
            out_sum += reconstructed[idx] * self.weight[idx]
        out_sum = out_sum / sum(self.weight)

        # store partial results for loss computation
        self.reconstructed = reconstructed

        # residual connection
        return out_sum + x

    @staticmethod
    def weights_init_kaiming(m):
        class_name = m.__class__.__name__
        if 'Linear' in class_name:
            nn.init.kaiming_normal(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif 'Conv2d' in class_name:
            nn.init.kaiming_normal(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif 'ConvTranspose2d' in class_name:
            nn.init.kaiming_normal(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif 'Norm' in class_name:
            m.weight.data.normal_(1.0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

    def weight_init(self):
        for m in self._modules:
            Net.weights_init_kaiming(m)
