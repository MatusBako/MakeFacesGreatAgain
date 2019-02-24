from math import log2, floor
from torch import nn, cat, add, ones, sum, mul
import torch.nn.init as init
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self, upscale_factor, num_channels=3, base_channel=64, s=12, m=4):
        super(Net, self).__init__()

        # Feature extraction
        self.num_recursions = 8
        # embedding layer
        self.embedding_layer = nn.Sequential(
            nn.Conv2d(num_channels, base_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channel, base_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # conv block of inference layer
        self.conv_block = nn.Sequential(nn.Conv2d(base_channel, base_channel, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True))

        # reconstruction layer
        self.reconstruction_layer = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(base_channel, num_channels, kernel_size=3, stride=1, padding=1)
        )

        # initial w
        self.w_init = ones(self.num_recursions) / self.num_recursions
        self.w = Variable(self.w_init.cuda(), requires_grad=True)

        self.weight_init()

    def forward(self, x):
        # embedding layer
        h0 = self.embedding_layer(x)

        # recursions
        h = [h0]
        for d in range(self.num_recursions):
            h.append(self.conv_block(h[d]))

        y_d_ = []
        out_sum = 0
        for d in range(self.num_recursions):
            y_d_.append(self.reconstruction_layer(h[d + 1]))
            out_sum += mul(y_d_[d], self.w[d])
        out_sum = mul(out_sum, 1.0 / (sum(self.w)))

        # skip connection
        final_out = add(out_sum, x)

        return y_d_, final_out

    @staticmethod
    def weights_init_kaiming(m):
        class_name = m.__class__.__name__
        if class_name.find('Linear') != -1:
            nn.init.kaiming_normal(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif class_name.find('Conv2d') != -1:
            nn.init.kaiming_normal(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif class_name.find('ConvTranspose2d') != -1:
            nn.init.kaiming_normal(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif class_name.find('Norm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

    def weight_init(self):
        for m in self._modules:
            Net.weights_init_kaiming(m)
