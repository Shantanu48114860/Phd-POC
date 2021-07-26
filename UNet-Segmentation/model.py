import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down sampling
        for feature in features:
            self.downs.append(DoubleConv(in_channels=in_channels, out_channels=feature))
            in_channels = feature

        # Up sampling
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(in_channels=feature * 2, out_channels=feature,
                                               kernel_size=2, stride=2))
            self.ups.append(DoubleConv(in_channels=feature * 2, out_channels=feature))

        self.bottleneck = DoubleConv(in_channels=features[-1], out_channels=features[-1] * 2)
        self.output = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1,
                                stride=1)

    def forward(self, x):
        # x: B x C x H x W
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections.reverse()
        for index, up in enumerate(self.ups):
            x = up(x)
            if index % 2 == 0:
                skip = skip_connections[index // 2]
                x = torch.cat((x, skip), dim=1)

        x = self.output(x)
        return x

if __name__=="__main__":
    model = UNet(in_channels=3, out_channels=1)
    x = torch.randn((2, 3, 32, 32))
    print(x.size())
    y_hat = model(x)
    print(y_hat.size())
    writer = SummaryWriter('runs/Models')
    writer.add_graph(model, x)
    writer.close()
