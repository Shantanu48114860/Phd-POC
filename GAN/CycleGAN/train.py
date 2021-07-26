import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

import config
from dataset import HorseZebraDataset
from discriminator_model import Discriminator
from generator_model import Generator
from utils import save_checkpoint, load_checkpoint


def train_fn(disc_H,
             disc_Z,
             gen_Z,
             gen_H,
             loader,
             opt_disc,
             opt_gen,
             L1,
             mse,
             d_scalar,
             g_scalar):
    loop = tqdm(loader, leave=True)
    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        # Train Discriminators H, Z
        with torch.cuda.amp.autocast():
            fake_horse = gen_H(zebra)
            D_H_real = disc_H(horse)
            D_H_Fake = disc_H(fake_horse.detach())
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_Fake, torch.zeros_like(D_H_Fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_zebra = gen_Z(horse)
            D_Z_real = disc_Z(zebra)
            D_Z_Fake = disc_Z(fake_zebra.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_Fake, torch.zeros_like(D_Z_Fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # total Disc loss
            D_loss = (D_H_loss + D_Z_loss) * 0.5

        opt_disc.zero_grad()
        d_scalar.scale(D_loss).backward()
        d_scalar.step(opt_disc)
        d_scalar.update()

        # Train Generators H, Z
        with torch.cuda.amp.autocast():
            # GAN loss for generators
            D_H_Fake = disc_H(fake_horse)
            D_Z_Fake = disc_Z(fake_zebra)
            loss_G_H = mse(D_H_Fake, torch.ones_like(D_H_Fake))
            loss_G_Z = mse(D_Z_Fake, torch.ones_like(D_Z_Fake))

            # cycle loss
            cycle_zebra = gen_Z(fake_horse)
            cycle_horse = gen_H(fake_zebra)
            cycle_zebra_loss = L1(zebra, cycle_zebra)
            cycle_horse_loss = L1(horse, cycle_horse)

            # identity loss(not used in horse/zebra dataset)
            identity_zebra = gen_Z(zebra)
            identity_horse = gen_H(horse)
            identity_zebra_loss = L1(zebra, identity_zebra)
            identity_horse_loss = L1(horse, identity_horse)

            G_loss = (
                    loss_G_Z +
                    loss_G_H +
                    config.LAMBDA_CYCLE * cycle_zebra_loss +
                    config.LAMBDA_CYCLE * cycle_horse_loss +
                    config.LAMBDA_IDENTITY * identity_zebra_loss +
                    config.LAMBDA_IDENTITY * identity_horse_loss
            )

        opt_gen.zero_grad()
        g_scalar.scale(G_loss).backward()
        g_scalar.step(opt_gen)
        g_scalar.update()

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

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE
        )

        load_checkpoint(
            config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE
        )

        load_checkpoint(
            config.CHECKPOINT_CRITIC_H, disc_H, opt_disc, config.LEARNING_RATE
        )

        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z, disc_Z, opt_disc, config.LEARNING_RATE
        )

    dataset = HorseZebraDataset(
        root_horse="../data/horse2zebra/horse2zebra/trainA",
        root_zebra="../data/horse2zebra/horse2zebra/trainB",
        transform=config.transforms
    )

    val_dataset = HorseZebraDataset(
        root_horse="../data/horse2zebra/horse2zebra/testA",
        root_zebra="../data/horse2zebra/horse2zebra/testB",
        transform=config.transforms
    )

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    g_scalar = torch.cuda.amp.GradScaler()
    d_scalar = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_H,
                 disc_Z,
                 gen_Z,
                 gen_H,
                 loader,
                 opt_disc,
                 opt_gen,
                 L1,
                 mse,
                 d_scalar,
                 g_scalar)

        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_H, opt_gen, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_Z, opt_gen, filename=config.CHECKPOINT_CRITIC_Z)


if __name__ == '__main__':
    main()
