import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import config
from dataset import HorseZebraDataset
from discriminator_model import Discriminator
from generator_model import Generator
from utils import load_checkpoint


def test_fn(
        gen_Z,
        gen_H,
        loader,
):
    gen_Z.eval()
    gen_H.eval()
    for idx, (zebra, horse) in enumerate(loader, 0):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)


        fake_horse = gen_H(zebra)
        fake_zebra = gen_Z(horse)

        if idx % 200 == 0:
            save_image(fake_horse * 0.5 + 0.5, f"saved_images/fake_horse_{idx}.png")
            save_image(horse * 0.5 + 0.5, f"saved_images/real_horse_{idx}.png")
            save_image(fake_zebra * 0.5 + 0.5, f"saved_images/fake_zebra_{idx}.png")
            save_image(zebra * 0.5 + 0.5, f"saved_images/real_zebra_{idx}.png")


def main():
    print("Cycle GAN ==>")
    print(config.DEVICE)
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    opt_gen = optim.Adam(
        list(gen_H.parameters()) + list(gen_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )
    print("Before loading")
    print(list(gen_H.parameters())[0][0][0])
    load_checkpoint(
        config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE
    )

    print("After loading")
    print(list(gen_H.parameters())[0][0][0])
    load_checkpoint(
        config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE
    )

    load_checkpoint(
        config.CHECKPOINT_CRITIC_H, disc_H, opt_disc, config.LEARNING_RATE
    )

    load_checkpoint(
        config.CHECKPOINT_CRITIC_Z, disc_Z, opt_disc, config.LEARNING_RATE
    )

    val_dataset = HorseZebraDataset(
        root_horse="../data/horse2zebra/horse2zebra/testA",
        root_zebra="../data/horse2zebra/horse2zebra/testB",
        transform=config.transforms
    )

    loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    g_scalar = torch.cuda.amp.GradScaler()
    d_scalar = torch.cuda.amp.GradScaler()

    test_fn(
        gen_Z,
        gen_H,
        loader
    )


if __name__ == '__main__':
    # main()
    import numpy as np

    a = torch.from_numpy(np.array([[1,2],[3,4]]))



    # Add by line and keep their 2D characteristics.

    print(torch.max(a, dim=1))



    # Add by line, do not maintain its two-dimensional characteristics

    print(torch.max(a, dim=0, keepdim=True))
