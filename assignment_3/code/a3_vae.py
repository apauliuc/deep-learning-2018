import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from torchvision.utils import make_grid, save_image

from datasets.bmnist import bmnist


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super(Encoder, self).__init__()

        self.input_dim = 784

        self.linear_hidden = nn.Linear(self.input_dim, hidden_dim)
        self.linear_mu = nn.Linear(hidden_dim, z_dim)
        self.linear_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        if input.shape[1] != self.input_dim:
            input = input.view(-1, self.input_dim)

        input = torch.tanh(self.linear_hidden(input))

        return self.linear_mu(input), self.linear_logvar(input)


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super(Decoder, self).__init__()

        self.linear_hidden = nn.Linear(z_dim, hidden_dim)
        self.linear_output = nn.Linear(hidden_dim, 784)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        input = torch.tanh(self.linear_hidden(input))
        return torch.sigmoid(self.linear_output(input))


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.zdim = z_dim

        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        input = input.view(input.shape[0], -1)
        # encoder
        mu, logvar = self.encoder(input)
        # reparameterization
        z = torch.add(mu, torch.mul(torch.sqrt(logvar.exp()), torch.randn_like(logvar)))
        # decoder
        output = self.decoder(z)

        # reconstruction loss
        recon_loss = F.binary_cross_entropy(output, input, reduction='sum')
        # regularization loss
        reg_loss = 0.5 * torch.sum(-logvar + logvar.exp() + mu.pow(2) - 1)
        # add losses
        average_negative_elbo = (recon_loss + reg_loss) / input.shape[0]

        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        # get sample, pass through decoder to get means, and sample images from bernoulli
        sample = torch.randn(n_samples, self.zdim)
        im_means = self.decoder(sample)
        sampled_ims = torch.bernoulli(im_means)
        return sampled_ims, im_means


def epoch_iter(model, data_loader, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    total_epoch_elbo = 0

    for batch_idx, data in enumerate(data_loader):
        optimizer.zero_grad()
        elbo = model(data)

        if model.training:
            elbo.backward()
            optimizer.step()

        total_epoch_elbo += (elbo.data * data.shape[0])

    average_epoch_elbo = total_epoch_elbo / len(data_loader.dataset)

    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_samples(sampled_ims, im_means, epoch, path):
    sampled_ims_grid = make_grid(sampled_ims.view(-1, 1, 28, 28), nrow=6, normalize=True)
    im_means_grid = make_grid(im_means.view(-1, 1, 28, 28), nrow=6, normalize=True)

    plt.figure()
    plt.imshow(sampled_ims_grid[0], cmap='gray')
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(os.path.join(path, 'images_epoch_' + str(epoch) + '.png'))
    plt.close()

    plt.figure()
    plt.imshow(im_means_grid[0].detach().numpy(), cmap='gray')
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(os.path.join(path, 'means_epoch_' + str(epoch) + '.png'))
    plt.close()


def main():
    save_folder = "output_vae"
    os.makedirs(save_folder, exist_ok=True)

    samples_count = 30

    data = bmnist()[:2]  # ignore test split
    model = VAE(hidden_dim=500, z_dim=ARGS.zdim)
    optimizer = torch.optim.Adam(model.parameters())

    sampled_ims, im_means = model.sample(samples_count)
    plot_samples(sampled_ims, im_means, '-1', save_folder)

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        train_elbo, val_elbo = run_epoch(model, data, optimizer)
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch+1}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functionality that is already imported.
        # --------------------------------------------------------------------

        if epoch in [int(ARGS.epochs/2), ARGS.epochs-1]:
            sampled_ims, im_means = model.sample(samples_count)

            plot_samples(sampled_ims, im_means, epoch+1, save_folder)

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------
    if ARGS.zdim == 2:
        steps = 20
        inverse_cdf = norm.ppf(np.linspace(1e-3, 1-1e-3, steps))
        x, y = np.meshgrid(inverse_cdf, inverse_cdf)
        z = torch.tensor([x.flatten(), y.flatten()]).float().t()
        manifold = model.decoder(z)
        save_image(manifold.view(-1, 1, 28, 28),
                   os.path.join(save_folder, 'manifold.png'),
                   nrow=steps, normalize=True)

    save_elbo_plot(train_curve, val_curve, os.path.join(save_folder, 'elbo.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
