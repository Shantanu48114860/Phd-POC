# import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Dataset_Custom import CustomDataset
from loss import NCELoss
from model import SIM_CLR


# def img_show(img):
#     plt.imshow(img.permute(1, 2, 0))
#     plt.show()


def save_model(model, out):
    torch.save(model.state_dict(), out)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

EPOCHS = 100
LEARNING_RATE = 0.0003
PROJECTION_DIM = 128
BATCH_SIZE = 64
TARGET_SIZE = 224
DATA_ROOT = "../data/STL10"
TEMPERATURE = 0.5
WEIGHT_DECAY = 1e-4

writer = SummaryWriter(f"logs")

train_dataset = torchvision.datasets.STL10(
    root=DATA_ROOT,
    split="unlabeled",
    download=False
)

ds = CustomDataset(train_dataset, target_size=TARGET_SIZE)
# idx = 123
# img_show(ds[123][0])
# img_show(ds[123][1])

loader = DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    shuffle=True
)

model = SIM_CLR(projection_dim=PROJECTION_DIM).to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

nce_loss = NCELoss(
    device=device,
    temperature=TEMPERATURE,
    batch_size=BATCH_SIZE
)

print(f"Size of the loader: {len(loader)}")
for epoch in range(EPOCHS):
    loss_epoch = 0
    model.train()
    for batch_idx, (x_i, x_j) in enumerate(loader):
        x_i = x_i.to(device)
        x_j = x_j.to(device)
        h_i, h_j, z_i, z_j = model(x_i, x_j)
        loss = nce_loss(z_i, z_j)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
        # print(f"Epoch: {epoch}, Loss: {loss_epoch / len(loader)}")
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss_epoch / len(loader)}")
        save_model(model, f"sim-clr_{epoch}.pth")

    writer.add_scalar("Loss/train", loss_epoch / len(loader), epoch)

save_model(model, "sim-clr.pth")
