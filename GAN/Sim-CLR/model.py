# paste this at the start of code
import ssl

import torch.nn as nn
import torchvision

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SIM_CLR(nn.Module):
    def __init__(self, projection_dim=64):
        super(SIM_CLR, self).__init__()
        self.encoder = torchvision.models.resnet18(pretrained=True)
        features_dim = self.encoder.fc.in_features
        self.encoder.fc = Identity()

        self.projection_heads = nn.Sequential(
            nn.Linear(
                in_features=features_dim,
                out_features=features_dim,
                bias=False
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=features_dim,
                out_features=projection_dim,
                bias=False
            )
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projection_heads(h_i)
        z_j = self.projection_heads(h_j)

        return h_i, h_j, z_i, z_j

# model = SIM_CLR(projection_dim=64)
# x_i = torch.randn(1, 3, 224, 224)
# x_j = torch.randn(1, 3, 224, 224)
# z_i, z_j, h_i, h_j = model(x_i, x_j)
# print(z_i.shape)
