import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, image_channels, discriminator_filters):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(image_channels, discriminator_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(discriminator_filters, discriminator_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(discriminator_filters * 2, discriminator_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(discriminator_filters * 4, discriminator_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(discriminator_filters * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)