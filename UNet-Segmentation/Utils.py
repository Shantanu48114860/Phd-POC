import torch
import torchvision
from torch.utils.data import DataLoader

from dataset import CarvanaDataset


def save_checkpoint(state, file_name="my_checkpoint.pth.tar"):
    print("=> save checkpoint")
    torch.save(state, file_name)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
        train_dir,
        train_mask_dir,
        val_dir,
        val_mask_dir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True
):
    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_mask_dir,
        transform=train_transform
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True)

    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_mask_dir,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixel = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixel += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
        print(
            "Got {0} with accuracy {1}".format(
                num_correct / num_pixel,
                num_correct / num_pixel * 100)
        )
        print("Dice score: {0}".format(dice_score / len(loader)))

    model.train()


def save_preds_img(
        loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(
            y.unsqueeze(1), f"{folder}/mask_{idx}.png"
        )
    model.train()
