import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        # Input: N x channels_img x 64 x 64
        self.disc = nn.Sequential(
            nn.Conv2d(
                in_channels=channels_img,
                out_channels=features_d,
                kernel_size=4,
                stride=2,
                padding=1
            ),  # 32 x 32
            nn.LeakyReLU(0.2),
            self._block(
                in_channels=features_d,
                out_channels=features_d * 2,
                kernel_size=4,
                stride=2,
                padding=1),  # 16 x 16
            self._block(
                in_channels=features_d * 2,
                out_channels=features_d * 4,
                kernel_size=4,
                stride=2,
                padding=1),  # 8 x 8
            self._block(
                in_channels=features_d * 4,
                out_channels=features_d * 8,
                kernel_size=4,
                stride=2,
                padding=1),  # 4 x 4
            nn.Conv2d(
                in_channels=features_d * 8,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=0
            ),  # 1 x1
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        # Input: N x z_dim x 1 x 1
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(
                in_channels=z_dim,
                out_channels=features_g * 16,
                kernel_size=4,
                stride=1,
                padding=0
            ),  # N x f_g x 4 x 4
            self._block(
                in_channels=features_g * 16,
                out_channels=features_g * 8,
                kernel_size=4,
                stride=2,
                padding=1
            ),  # N x f_g x 8 x 8
            self._block(
                in_channels=features_g * 8,
                out_channels=features_g * 4,
                kernel_size=4,
                stride=2,
                padding=1
            ),  # N x f_g x 16 x 16
            self._block(
                in_channels=features_g * 4,
                out_channels=features_g * 2,
                kernel_size=4,
                stride=2,
                padding=1
            ),  # N x f_g x 32 x 32
            nn.ConvTranspose2d(
                in_channels=features_g * 2,
                out_channels=channels_img,
                kernel_size=4,
                stride=2,
                padding=1
            ),  # 64 x 64
            nn.Tanh()  # [-1, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.gen(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn(N, in_channels, H, W)
    disc = Discriminator(in_channels, features_d=8)
    initialize_weights(disc)
    print(disc(x).shape)
    assert disc(x).shape == (N, 1, 1, 1)

    gen = Generator(z_dim=z_dim, channels_img=3, features_g=8)
    initialize_weights(gen)
    z = torch.randn(N, z_dim, 1, 1)
    print(z.shape)
    print(gen(z).shape)
    assert gen(z).shape == (N, 3, 64, 64)


# test()
