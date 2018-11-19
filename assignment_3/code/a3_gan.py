import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.layer_seq = nn.Sequential(
            nn.Linear(args.latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.layer_seq(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layer_seq = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.layer_seq(img).view(-1)


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, path):
    criterion = nn.BCELoss()

    device = torch.device("cpu")
    if torch.cuda.is_available():
        discriminator.cuda()
        generator.cuda()
        criterion.cuda()
        device = torch.device("cuda:0")

    fixed_noise = torch.randn(args.batch_size, args.latent_dim, device=device)

    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            # Create labels, noise and generated image from noise
            true_target = torch.ones(imgs.shape[0], device=device)
            fake_target = torch.zeros(imgs.shape[0], device=device)
            noise = torch.randn(imgs.shape[0], args.latent_dim, device=device)
            generated_img = generator(noise)

            # Train Generator
            # ---------------
            generator.zero_grad()
            d_output = discriminator(generated_img)
            g_loss = criterion(d_output, true_target)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            # -------------------
            for d_step in range(args.discriminator_steps):
                discriminator.zero_grad()

                # compute loss on real data
                d_real_output = discriminator(imgs.view(imgs.shape[0], -1).to(device))
                d_real_loss = criterion(d_real_output, true_target)

                # compute loss on generated data
                d_fake_output = discriminator(generated_img.detach())
                d_fake_loss = criterion(d_fake_output, fake_target)

                # add losses and proceed with backward step
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()

            # print statistics
            if i % 100 == 0:
                print('Epoch [%d/%d] Progress [%d/%d] Loss_D: %.4f Loss_G: %.4f'
                      % (epoch, args.n_epochs, i, len(dataloader),
                         d_loss, g_loss))

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0 and batches_done != 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                save_image(generated_img[:25].view(-1, 1, 28, 28),
                           path + '/random_{}.png'.format(batches_done),
                           nrow=5, normalize=True)


def main():
    # Create output image directory
    path = "output_gan"
    os.makedirs(path, exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator()
    discriminator = Discriminator()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D, path)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    generator.cpu()
    torch.save(generator.state_dict(), path + "/gan_k_{}.pt".format(args.discriminator_steps))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--discriminator_steps', type=int, default=1,
                        help='train discriminator k steps for 1 generator step')
    args = parser.parse_args()

    main()
