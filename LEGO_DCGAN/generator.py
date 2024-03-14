import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise, generator_filters, image_channels):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(noise, generator_filters * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(generator_filters * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(generator_filters * 8, generator_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_filters * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(generator_filters * 4, generator_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_filters * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(generator_filters * 2, generator_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_filters),
            nn.ReLU(True),
            nn.ConvTranspose2d(generator_filters, image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)