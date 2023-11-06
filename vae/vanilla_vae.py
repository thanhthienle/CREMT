import torch
from vae import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
from tqdm import tqdm


class VanillaVAE(BaseVAE):
    def __init__(self, args,
                 in_channels: int = 1536,
                 latent_dim: int = 512,
                 hidden_dims: List = [],
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        if not hidden_dims:
            hidden_dims = [768, 512]

        # Changing in_channels
        changing_in_channels = in_channels

        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.extend([
                nn.Linear(changing_in_channels, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(),
            ])
            changing_in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        self.device = args.device

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.LeakyReLU(),
            ])

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Linear(in_features=hidden_dims[-1], out_features=hidden_dims[-1]),
                            nn.BatchNorm1d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Linear(in_features=hidden_dims[-1], out_features=in_channels),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        print(f"ENCODER: {self.encoder[0][0].weight.requires_grad}")
        print(f"ENCODER: {self.encoder[1][0].weight.requires_grad}")
        print(f"result: {result.requires_grad}")
        print("")
        # result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        # result = result.view(-1, 512, 4)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device=None, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        if current_device is None:
            current_device = self.device

        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
    
    def fit(self, data_loader: Tensor, epochs: int, learning_rate):
        """
        Given a dataset data, returns a trained VAE object
        :param data_loader: (Tensor) [N x B x 1536]
        """
        optimizer = torch.optim.Adam([
            dict(params=self.parameters(), lr=learning_rate),
        ])
        for i in range(epochs):
            td = tqdm(data_loader, desc=f"Train VAE epoch {i+1}/{epochs}")
            for (_, tokens, _) in td:
                tokens = torch.stack([x.to(self.device) for x in tokens], dim=0) # GPU
                optimizer.zero_grad()
                tokens_hat, mu, sigma = self.forward(tokens)
                print(tokens_hat.requires_grad)
                print(mu.requires_grad)
                print(sigma.requires_grad)
                sigma = torch.exp(sigma)
                print(sigma.requires_grad)
                loss = ((tokens - tokens_hat)**2).sum() + (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
                loss.backward()
                optimizer.step()
                td.set_postfix(loss=loss)
