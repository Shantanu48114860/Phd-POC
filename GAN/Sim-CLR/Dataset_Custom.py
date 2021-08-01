import torchvision
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, ds, target_size):
        self.ds = ds
        self.target_size = target_size
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=self.target_size),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, _ = self.ds[idx]
        return self.train_transform(image), self.train_transform(image)
