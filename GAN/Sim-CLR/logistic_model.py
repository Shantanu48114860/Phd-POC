import torch
import torch.nn as nn
import torchvision

from model import SIM_CLR


class Logistic_Model(nn.Module):
    def __init__(self, device, projection_dim=128,
                 n_classes=10, path=None):
        super(Logistic_Model, self).__init__()
        self.backbone = SIM_CLR(projection_dim=projection_dim)
        self.backbone.load_state_dict(
            torch.load(path,
                       map_location=device)
        )
        n_features = self.backbone.projection_heads[0].out_features
        self.backbone = self.backbone.to(device)
        self.backbone.eval()
        self.linear = nn.Linear(in_features=n_features,
                                out_features=n_classes)
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        h, _, z, _ = self.backbone(x, x)
        return self.linear(h)


# device = torch.device("cuda" if torch.cuda.is_available()
#                       else "cpu")
# IMAGE_SIZE = 224
# DATA_ROOT = "../data/STL10"
# EPOCHS = 100
# LEARNING_RATE = 0.0003
# PROJECTION_DIM = 128
# BATCH_SIZE = 3
# TEMPERATURE = 0.5
# WEIGHT_DECAY = 1e-4
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
#
# transforms = torchvision.transforms.Compose([
#     torchvision.transforms.Resize(IMAGE_SIZE),
#     torchvision.transforms.ToTensor()
# ])
# train_dataset = torchvision.datasets.STL10(
#     root=DATA_ROOT,
#     split="train",
#     download=False,
#     transform=transforms
# )
#
# train_loader = torch.utils.data.DataLoader(
#     train_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     drop_last=True,
# )
#
#
# dataloader_iterator = iter(train_loader)
# print(device)
# # x = torch.randn(1, 3, 224, 224)
# x, y = next(dataloader_iterator)
# print(x.shape)
# lr = Logistic_Model(device)
# optimizer = torch.optim.Adam(lr.parameters(), lr=3e-4)
# print("###Grad####")
# print(next(lr.backbone.parameters())[0][0][0])
# print(next(lr.linear.parameters())[0][0:10])
# print("###Grad####")
# output = lr(x)
# print(output)
# # print(output)
# print(output.shape)
# criterion = torch.nn.CrossEntropyLoss()
# target = torch.randint(0, 2, (1,))
# loss = criterion(output, y)
# optimizer.zero_grad()
# loss.backward()
# print("###Grad####")
# print(next(lr.backbone.parameters())[0][0][0])
# print(next(lr.linear.parameters())[0][0:10])
# print("###Grad####")
# # print(output.data)
# print(loss)
# optimizer.step()