import torch
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Discriminator, Generator, initialize_weights
from utils import gradient_penalty

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 25
FEATURES_DISC = 64
FEATURES_GEN = 64
GENERATE_EMBEDDINGS = Z_DIM
CRITIC_ITERATIONS = 5
LAMBDA = 10
NUM_CLASSES = 10

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)],
            [0.5 for _ in range(CHANNELS_IMG)]
        )
    ]
)

dataset = datasets.MNIST(root="../data/MNIST", train=True, transform=transforms,
                         download=True)

# dataset = datasets.ImageFolder(root="celebA", transform=transforms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(
    z_dim=Z_DIM,
    channels_img=CHANNELS_IMG,
    features_g=FEATURES_GEN,
    num_classes=NUM_CLASSES,
    img_size=IMAGE_SIZE,
    embed_size=GENERATE_EMBEDDINGS
).to(device)
disc = Discriminator(
    channels_img=CHANNELS_IMG,
    features_d=FEATURES_DISC,
    num_classes=NUM_CLASSES,
    img_size=IMAGE_SIZE
).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0, 0.9))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0, 0.9))

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, labels) in enumerate(loader):
        real = real.to(device)
        labels = labels.to(device)
        # train critic/discriminator
        loss_critic = 0
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(real.shape[0], Z_DIM, 1, 1).to(device)
            fake = gen(noise, labels)
            critic_real = disc(real, labels).reshape(-1)
            critic_fake = disc(fake, labels).reshape(-1)
            gp = gradient_penalty(disc, labels, real, fake, device=device)
            loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + LAMBDA * gp
            )
            disc.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_disc.step()

        # train generator min(G) = -E(D(G(z))
        output_g = disc(fake, labels)
        loss_gen = -torch.mean(output_g)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # print loss
        if batch_idx % 100 == 0:
            gen.eval()
            disc.eval()
            print(
                f"Epoch [{epoch}/ {NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} "
                f"Loss D: {loss_critic: .4f}, Loss G: {loss_gen: .4f}"
            )

            with torch.no_grad():
                # log to tensorboard for 32 samples
                fake = gen(noise, labels)
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )
                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_real.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
            gen.train()
            disc.train()
