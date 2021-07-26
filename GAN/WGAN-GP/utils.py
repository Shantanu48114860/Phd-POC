import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import Discriminator, Generator, initialize_weights


def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand(BATCH_SIZE, 1, 1, 1)
    # print(real.shape)
    # print(fake.shape)
    # print(epsilon.shape)
    # print(epsilon)
    epsilon = epsilon.repeat(1, C, H, W).to(device)
    # print(epsilon.shape)
    # print(epsilon)


    interpolated_images = real * epsilon + fake * (1 - epsilon)
    # calculate critic scores
    mixed_scores = critic(interpolated_images)
    # print(mixed_scores.shape)
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]


    # print(interpolated_images.shape)
    # print(gradient.shape)

    gradient = gradient.view(gradient.shape[0], -1)
    # print(gradient.shape)

    # L2 norm
    gradient_norm = gradient.norm(2, dim=1)
    # print(gradient_norm.shape)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    # print(gradient_penalty)
    return gradient_penalty

def save_checkpoint(state, filename="celeba_wgan_gp.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, gen, disc):
    print("=> Loading checkpoint")
    gen.load_state_dict(checkpoint['gen'])
    disc.load_state_dict(checkpoint['disc'])

# def test():
    # IMAGE_SIZE = 64
    # CHANNELS_IMG = 1
    # gen = Generator(z_dim=100, channels_img=1, features_g=8)
    # critic = Discriminator(channels_img=1, features_d=8)
    # initialize_weights(gen)
    # initialize_weights(critic)
    # transforms = transforms.Compose(
    #     [
    #         transforms.Resize(IMAGE_SIZE),
    #         transforms.ToTensor(),
    #         transforms.Normalize(
    #             [0.5 for _ in range(CHANNELS_IMG)],
    #             [0.5 for _ in range(CHANNELS_IMG)]
    #         )
    #     ]
    # )
    #
    # dataset = datasets.MNIST(root="../data/MNIST", train=True, transform=transforms,
    #                          download=True)
    # loader = DataLoader(dataset, batch_size=2, shuffle=True)
    # for batch_idx, (real, _) in enumerate(loader):
    #     noise = torch.randn(real.shape[0], 100, 1, 1)
    #     fake = gen(noise)
    #     gradient_penalty(critic, real, fake)
    #     break


# test()
