import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 down=True,
                 use_act=True,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 output_padding=0
                 ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                padding_mode="reflect",
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ) if down else
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1, stride=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1, stride=1)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels=img_channels,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect"
            ),
            nn.ReLU(inplace=True)
        )

        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=num_features,
                    out_channels=num_features * 2,
                    down=True,
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                ConvBlock(
                    in_channels=num_features * 2,
                    out_channels=num_features * 4,
                    down=True,
                    kernel_size=3,
                    stride=2,
                    padding=1
                )
            ]
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )

        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=num_features * 4,
                    out_channels=num_features * 2,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),
                ConvBlock(
                    in_channels=num_features * 2,
                    out_channels=num_features,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                )
            ]
        )

        self.last = nn.Conv2d(
            in_channels=num_features,
            out_channels=3,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect"
        )

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)

        x = self.residual_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)

        return torch.tanh(self.last(x))


class test():
    x = torch.randn(2, 3, 256, 256)
    print(x.shape)
    model = Generator(img_channels=3, num_residuals=9)
    pred = model(x)
    print(model)
    print(pred.shape)


if __name__ == '__main__':
    test()

