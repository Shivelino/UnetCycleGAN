import torch
from torch import nn
import torch.nn.functional as F
from .common_modules import ResidualBlock, DiscrimConv


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, output_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        self.disconv1 = DiscrimConv(input_nc, 64, 4, 2, 1, act=True, slope=0.2)
        self.disconv2 = DiscrimConv(64, 128, 4, 2, 1, act=True, slope=0.2)
        self.disconv3 = DiscrimConv(128, 256, 4, 2, 1, act=True, slope=0.2)
        self.disconv4 = DiscrimConv(256, 512, 4, 1, 1, act=True, slope=0.2)
        # FCN classification layer
        self.conv5 = nn.Conv2d(512, 1, 4, padding=1)

    def forward(self, x):
        x = self.disconv1.forward_nonorm(x)
        x = self.disconv2.forward(x)
        x = self.disconv3.forward(x)
        x = self.disconv4.forward(x)
        x = self.conv5.forward(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

